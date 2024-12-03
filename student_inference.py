import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from datetime import datetime
import time
import os

class CompactStudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CompactStudentNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class QuickDrawInference:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.setup_logging()
        
        self.classes = [
            'circle',        # 단순한 원
            'triangle',      # 삼각형
            'square',        # 사각형
            'donut',         # 초승달
            'house',      # 마름모
            'cloud',      # 팔각형
            'lightning',    # 번개
            'star',         # 별
            'diamond',      # 육각형
            'banana'          # 직선
        ]
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.metrics = {
            'fps': [],
            'inference_times': []
        }

    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename=f'logs/improved_inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CompactStudentNet(num_classes=len(self.classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

# preprocess_frame 메소드 부분만 수정된 코드입니다
    def preprocess_frame(self, frame):
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 적응형 이진화 적용
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 노이즈 제거
        binary = cv2.medianBlur(binary, 3)
        
        # 윤곽선 검출 및 ROI 추출
        contours_output = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_output[0] if len(contours_output) == 2 else contours_output[1]
        
        if contours:
            # 가장 큰 윤곽선 찾기
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 여백 추가
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary.shape[1] - x, w + 2*padding)
            h = min(binary.shape[0] - y, h + 2*padding)
            
            # ROI 추출
            roi = binary[y:y+h, x:x+w]
            
            # 정사각형으로 만들기
            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)
            start_x = (size - w) // 2
            start_y = (size - h) // 2
            square[start_y:start_y+h, start_x:start_x+w] = roi
            
            # 28x28로 리사이즈
            resized = cv2.resize(square, (28, 28))
        else:
            resized = np.zeros((28, 28), dtype=np.uint8)
        
        # 정규화 및 텐서 변환
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        
        # 시각화를 위한 전처리 단계별 이미지 반환
        return tensor, {
            'gray': gray,
            'binary': binary,
            'resized': resized
        }

    def run_inference(self, use_usb=False, duration=60):
        if use_usb:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)640, height=(int)480, "
                "format=(string)NV12, framerate=(fraction)30/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, width=(int)640, height=(int)480, "
                "format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink",
                cv2.CAP_GSTREAMER
            )

        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            return

        start_time = time.time()
        frames_processed = 0

        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                inference_start = time.time()
                tensor, preprocess_images = self.preprocess_frame(frame)
                
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                inference_time = (time.time() - inference_start) * 1000
                fps = 1000 / inference_time if inference_time > 0 else 0
                
                # 결과 표시
                confidence = conf.item() * 100
                predicted_class = self.classes[pred.item()]
                color = (0, 255, 0) if confidence > 50 else (0, 0, 255)

                # 28x28 이미지를 좀 더 크게 확대 (140x140)
                input_display = cv2.resize(preprocess_images['resized'], (140, 140))
                input_display = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)

                # 원본 프레임의 좌상단에 입력 이미지 표시
                y_offset = 10
                x_offset = 10
                
                # 흰색 배경 추가
                cv2.rectangle(frame, (x_offset-5, y_offset-5),
                            (x_offset+145, y_offset+145), (255,255,255), -1)
                
                # 테두리 추가
                cv2.rectangle(frame, (x_offset-5, y_offset-5),
                            (x_offset+145, y_offset+145), (0,0,0), 1)
                
                # 입력 이미지 표시
                frame[y_offset:y_offset+140, x_offset:x_offset+140] = input_display

                # 결과 텍스트를 입력 이미지 오른쪽에 표시
                text_x = x_offset + 160
                cv2.putText(frame, f"Class: {predicted_class}", (text_x, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Conf: {confidence:.1f}%", (text_x, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Top-3 예측 표시
                probs_np = probs.squeeze().cpu().numpy()
                top3_idx = probs_np.argsort()[-3:][::-1]
                for i, idx in enumerate(top3_idx):
                    text = f"#{i+1}: {self.classes[idx]} ({probs_np[idx]*100:.1f}%)"
                    cv2.putText(frame, text, (text_x, 140 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('QuickDraw Inference', frame)

                self.metrics['fps'].append(fps)
                self.metrics['inference_times'].append(inference_time)
                frames_processed += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.logger.info("Inference interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_metrics(frames_processed)

    def save_metrics(self, frames_processed):
        if not self.metrics['fps']:
            return
            
        avg_fps = np.mean(self.metrics['fps'])
        avg_inference = np.mean(self.metrics['inference_times'])
        
        self.logger.info("\nInference Results:")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Average Inference Time: {avg_inference:.2f}ms")
        self.logger.info(f"Total Frames: {frames_processed}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained student model')
    parser.add_argument('--use-usb', action='store_true',
                      help='Use USB camera instead of CSI')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration in seconds')
    
    args = parser.parse_args()
    
    inference = QuickDrawInference(args.model)
    inference.run_inference(use_usb=args.use_usb, duration=args.duration)

if __name__ == '__main__':
    main()