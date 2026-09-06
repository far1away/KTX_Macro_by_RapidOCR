from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np
import subprocess
import time
import os
import requests

print("초고속 RapidOCR 엔진 로딩 중...")
REC_MODEL_PATH = "korean_PP-OCRv3_rec_infer.onnx"
KEYS_PATH = "korean_dict.txt"

if not os.path.exists(REC_MODEL_PATH) or not os.path.exists(KEYS_PATH):
    print(f"[오류] 모델 파일 또는 사전 파일이 없습니다.")
    exit()

ocr = RapidOCR(rec_model_path=REC_MODEL_PATH, keys_path=KEYS_PATH)
DEVICE_ID = "127.0.0.1:5555"
TARGET_KEYWORD = "결제할티켓"

# =================================================================
# [설정] 텔레그램 봇 정보 (변수에 값을 채워 넣으세요)
# =================================================================
TELEGRAM_TOKEN = ""  # 예: "123456789:ABCdefGhI..."
CHAT_ID = ""    # 예: "123456789"

def send_telegram_message(message):
    """텔레그램 메시지 전송 함수"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("\n[알림] 텔레그램 토큰 또는 Chat ID가 비어 있어 메시지를 전송하지 못했습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("\n📲 [알림] 텔레그램 메시지 전송 완료!")
        else:
            print(f"\n[오류] 텔레그램 전송 실패 (코드: {response.status_code})")
    except Exception as e:
        print(f"\n[오류] 텔레그램 통신 중 에러 발생: {e}")

subprocess.run(f"adb connect {DEVICE_ID}", shell=True, stdout=subprocess.DEVNULL)

def get_adb_screenshot():
    pipe = subprocess.Popen(
        f"adb -s {DEVICE_ID} exec-out screencap -p",
        stdout=subprocess.PIPE,
        shell=True
    )
    image_bytes = pipe.stdout.read()
    if not image_bytes: return None
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def send_adb_touch(x, y):
    subprocess.run(f"adb -s {DEVICE_ID} shell input tap {x} {y}", shell=True, stdout=subprocess.DEVNULL)

current_step = 1
step_start_time = time.time()  # 단계별 타이머 측정 시작

print("\n🚀 [타임아웃 적용 초고속 버전] 매크로 가동 시작!")
print("진행 순서: [1단계] 간편 예매 -> [2단계] 바로 예매 -> [3단계] Confirm(1단계 복귀) OR 결제할 티켓(성공)")
print("⏱️ 각 단계에서 60초 이상 반응이 없으면 자동으로 1단계로 리셋됩니다.\n")

# 매크로 시작 시 텔레그램 알림 전송
send_telegram_message("🚀 [매크로 시작] 예매 자동화 프로그램이 가동되었습니다. 모니터링을 시작합니다!")

try:
    while True:
        # 타임아웃 검사 (1단계가 아닌 상태에서 60초가 경과한 경우)
        if current_step > 1 and (time.time() - step_start_time > 60):
            print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ 60초 타임아웃 발생! 응답이 없어 1단계(간편 예매)로 복귀합니다.")
            current_step = 1
            step_start_time = time.time()

        frame = get_adb_screenshot()
        if frame is None:
            time.sleep(0.5)
            continue

        result, _ = ocr(frame)

        if result:
            # [1단계] '간편 예매'
            if current_step == 1:
                found = False
                for box, text, prob in result:
                    clean_text = text.replace(" ", "")
                    if prob > 0.2 and "간편예매" in clean_text:
                        cx = int((box[0][0] + box[2][0]) / 2)
                        cy = int((box[0][1] + box[2][1]) / 2)
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [1단계] '간편 예매' 터치! ({cx}, {cy})")
                        current_step = 2
                        step_start_time = time.time()  # 타이머 리셋
                        found = True
                        time.sleep(0.5)
                        break
                if not found:
                    print(f"[{time.strftime('%H:%M:%S')}] [1단계] 스캔 중...", end="\r")

            # [2단계] '바로 예매'
            elif current_step == 2:
                found = False
                for box, text, prob in result:
                    clean_text = text.replace(" ", "")
                    if prob > 0.2 and "바로예매" in clean_text:
                        cx = int((box[0][0] + box[2][0]) / 2)
                        cy = int((box[0][1] + box[2][1]) / 2)
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [2단계] '바로 예매' 터치! ({cx}, {cy})")
                        current_step = 3
                        step_start_time = time.time()  # 타이머 리셋
                        found = True
                        time.sleep(0.5)
                        break
                if not found:
                    print(f"[{time.strftime('%H:%M:%S')}] [2단계] 스캔 중 (대기 시간: {int(time.time() - step_start_time)}초)...", end="\r")

            # [3단계] 'Confirm' 또는 '결제할 티켓'
            elif current_step == 3:
                # 1) 성공 감지
                success = False
                for box, text, prob in result:
                    clean_text = text.replace(" ", "")
                    if prob > 0.2 and TARGET_KEYWORD in clean_text:
                        print("\n🎉 [성공!] '결제할 티켓' 진입 성공! 텔레그램 메시지를 전송합니다.")
                        send_telegram_message("🚨 [매크로 성공] 티켓 예매 화면('결제할 티켓') 진입 완료! 빨리 확인하세요!")
                        input("\n매크로를 종료하려면 Enter를 누르세요...")
                        exit()

                # 2) Confirm 감지
                found = False
                for box, text, prob in result:
                    clean_text = text.replace(" ", "").lower()
                    if prob > 0.2 and "confirm" in clean_text:
                        cx = int((box[0][0] + box[2][0]) / 2)
                        cy = int((box[0][1] + box[2][1]) / 2)
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [3단계] 'Confirm' 터치 -> 1단계 복귀")
                        current_step = 1
                        step_start_time = time.time()  # 타이머 리셋
                        found = True
                        time.sleep(0.5)
                        break
                if not found:
                    print(f"[{time.strftime('%H:%M:%S')}] [3단계] 스캔 중 (대기 시간: {int(time.time() - step_start_time)}초)...", end="\r")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n매크로가 종료되었습니다.")