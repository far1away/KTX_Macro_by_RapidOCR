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
TARGET_KEYWORD = "결제할티켓"
CONFIRM_TEXTS = ("confirm", "확인", "거후")

# =================================================================
# [설정] 텔레그램 봇 정보 (변수에 값을 채워 넣으세요)
# =================================================================
TELEGRAM_TOKEN = ""  # 예: "123456789:ABCdefGhI..."
CHAT_ID = ""    # 예: "123456789"
HEARTBEAT_INTERVAL = 30 * 60
STEP_TIMEOUT = 5

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

def get_connected_device():
    """연결된 에뮬레이터 또는 실제 휴대폰의 ADB ID를 자동 탐지 (호환성 개선 버전)"""
    try:
        process = subprocess.Popen("adb devices", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, _ = process.communicate()
        lines = stdout.strip().split("\n")[1:]
        devices = []
        for line in lines:
            if "\tdevice" in line:
                dev_id = line.split("\t")[0]
                devices.append(dev_id)
        
        if not devices:
            return None
        
        selected = devices[0]
        print(f"🔗 [자동 감지된 기기] {selected}")
        return selected
    except Exception as e:
        print(f"[오류] 기기 탐지 실패: {e}")
        return None

# 기기 자동 연결 시도 (LDPlayer 또는 실제 폰 공용)
DEVICE_ID = get_connected_device()
if not DEVICE_ID:
    print("\n[오류] 연결된 기기(에뮬레이터 또는 USB 휴대폰)를 찾을 수 없습니다.")
    print("👉 LDPlayer가 켜져 있거나, 휴대폰이 USB 디버깅 허용 상태로 연결되어 있는지 확인해주세요.")
    exit()

# IP 포트 형태(LDPlayer 등)인 경우 adb connect 시도
if ":" in DEVICE_ID:
    subprocess.run(f"adb connect {DEVICE_ID}", shell=True, stdout=subprocess.DEVNULL)

def set_screen_awake(enabled):
    state = "true" if enabled else "false"
    subprocess.run(
        ["adb", "-s", DEVICE_ID, "shell", "svc", "power", "stayon", state],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

set_screen_awake(True)

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

def normalize_ocr_text(text):
    """OCR 결과에서 공백과 구두점을 제거해 버튼 문구 비교를 안정화합니다."""
    return "".join(char for char in text.lower() if char.isalnum() or "가" <= char <= "힣")

def get_upscaled_ocr(frame, scale=2.0):
    """작은 휴대폰 버튼을 보완하기 위해 확대 이미지로 OCR을 수행합니다."""
    height, width = frame.shape[:2]
    upscaled = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    result, _ = ocr(upscaled)
    return result, scale

def find_blue_button(frame):
    """OCR이 실패한 경우 하단의 파란색 확인 버튼 중심 좌표를 찾습니다."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([145, 255, 255]))
    kernel_size = max(3, int(width * 0.01) | 1)
    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8),
    )
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, button_width, button_height = cv2.boundingRect(contour)
        area = button_width * button_height
        if (
            y > height * 0.55
            and area > width * height * 0.001
            and button_width > button_height * 2
            and button_width > width * 0.2
            and button_height < height * 0.2
        ):
            candidates.append((area, x, y, button_width, button_height))

    if not candidates:
        return None

    _, x, y, button_width, button_height = max(candidates)
    return x + button_width // 2, y + button_height // 2

def find_blue_ocr_button(frame, result):
    """OCR 오인식 결과 주변의 파란 배경을 확인해 버튼 글자 중심을 찾습니다."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([145, 255, 255]))

    for box, text, prob in result or []:
        clean_text = normalize_ocr_text(text)
        if prob <= 0.2 or not clean_text or len(clean_text) > 4:
            continue

        x1 = max(0, int(min(point[0] for point in box)))
        y1 = max(0, int(min(point[1] for point in box)))
        x2 = min(width, int(max(point[0] for point in box)))
        y2 = min(height, int(max(point[1] for point in box)))
        if y1 < height * 0.55 or x2 <= x1 or y2 <= y1:
            continue

        padding_x = max((x2 - x1) * 2, int(width * 0.03))
        padding_y = max((y2 - y1) * 2, int(height * 0.015))
        region_x1 = max(0, x1 - padding_x)
        region_y1 = max(0, y1 - padding_y)
        region_x2 = min(width, x2 + padding_x)
        region_y2 = min(height, y2 + padding_y)
        region = blue_mask[region_y1:region_y2, region_x1:region_x2]

        if region.size and cv2.countNonZero(region) / region.size > 0.08:
            return (x1 + x2) // 2, (y1 + y2) // 2

    return None

current_step = 1
step_start_time = time.time()  # 단계별 타이머 측정 시작
last_heartbeat_time = time.time()
timeout_recovery = False
timeout_recovery_step = 1

print("\n🚀 [크로스포맷/휴대폰 호환 버전] 매크로 가동 시작!")
print("진행 순서: [1단계] 간편 예매 -> [2단계] 바로 예매 -> [3단계] Confirm(1단계 복귀) OR 결제할 티켓(성공)")
print("⏱️ 각 단계에서 60초 이상 반응이 없으면 자동으로 1단계로 리셋됩니다.\n")

# 매크로 시작 시 텔레그램 알림 전송
send_telegram_message("🚀 [매크로 시작] 예매 자동화 프로그램이 가동되었습니다. 모니터링을 시작합니다!")

try:
    while True:
        if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
            send_telegram_message("💓 [매크로 하트비트] 프로그램이 정상적으로 실행 중입니다.")
            last_heartbeat_time = time.time()

        # 타임아웃 발생 시 현재 화면에서 보이는 단계부터 순서대로 재개합니다.
        if time.time() - step_start_time > STEP_TIMEOUT:
            print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ {STEP_TIMEOUT}초 타임아웃 발생! 화면에 보이는 단계부터 재시도합니다.")
            timeout_recovery_step = current_step
            step_start_time = time.time()
            timeout_recovery = True

        frame = get_adb_screenshot()
        if frame is None:
            time.sleep(0.5)
            continue

        result, _ = ocr(frame)

        if timeout_recovery and result:
            recovered_texts = [normalize_ocr_text(text) for _, text, _ in result]
            if timeout_recovery_step <= 1 and any("간편예매" in text for text in recovered_texts):
                current_step = 1
                timeout_recovery = False
            elif timeout_recovery_step <= 2 and any("바로예매" in text for text in recovered_texts):
                current_step = 2
                timeout_recovery = False
            elif (
                timeout_recovery_step <= 3
                and (
                any(TARGET_KEYWORD in text for text in recovered_texts)
                or any(text in CONFIRM_TEXTS for text in recovered_texts)
                or find_blue_ocr_button(frame, result)
                )
            ):
                current_step = 3
                timeout_recovery = False

        # OCR 결과가 비어도 파란색 확인 버튼 검사는 실행합니다.
        if current_step == 3 and not result:
            blue_button = find_blue_button(frame)
            if blue_button:
                cx, cy = blue_button
                send_adb_touch(cx, cy)
                print(f"[{time.strftime('%H:%M:%S')}] [3단계] 파란 확인 버튼 터치 -> 1단계 복귀 ({cx}, {cy})")
                current_step = 1
                step_start_time = time.time()
                time.sleep(0.5)
                continue

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
                        break
                if not found:
                    print(f"[{time.strftime('%H:%M:%S')}] [1단계] 스캔 중 (대기 시간: {int(time.time() - step_start_time)}초)...", end="\r")

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
                        set_screen_awake(False)
                        input("\n매크로를 종료하려면 Enter를 누르세요...")
                        exit()

                # 2) Confirm 감지
                found = False
                for box, text, prob in result:
                    clean_text = normalize_ocr_text(text)
                    if prob > 0.2 and clean_text in CONFIRM_TEXTS:
                        cx = int((box[0][0] + box[2][0]) / 2)
                        cy = int((box[0][1] + box[2][1]) / 2)
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [3단계] 'Confirm or 확인' 터치 -> 1단계 복귀")
                        current_step = 1
                        step_start_time = time.time()  # 타이머 리셋
                        found = True
                        break

                # 휴대폰에서는 확인 버튼이 작게 캡처될 수 있어 확대 이미지로 한 번 더 검사합니다.
                if not found:
                    enlarged_result, scale = get_upscaled_ocr(frame)
                    for box, text, prob in enlarged_result or []:
                        clean_text = normalize_ocr_text(text)
                        if prob > 0.2 and clean_text in CONFIRM_TEXTS:
                            cx = int(((box[0][0] + box[2][0]) / 2) / scale)
                            cy = int(((box[0][1] + box[2][1]) / 2) / scale)
                            send_adb_touch(cx, cy)
                            print(f"[{time.strftime('%H:%M:%S')}] [3단계] 확대 OCR로 'Confirm or 확인' 터치 -> 1단계 복귀")
                            current_step = 1
                            step_start_time = time.time()
                            found = True
                            break

                if not found:
                    blue_ocr_button = find_blue_ocr_button(frame, result)
                    if blue_ocr_button:
                        cx, cy = blue_ocr_button
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [3단계] OCR 주변 파란 확인 버튼 터치 -> 1단계 복귀 ({cx}, {cy})")
                        current_step = 1
                        step_start_time = time.time()
                        found = True

                if not found:
                    blue_button = find_blue_button(frame)
                    if blue_button:
                        cx, cy = blue_button
                        send_adb_touch(cx, cy)
                        print(f"[{time.strftime('%H:%M:%S')}] [3단계] 파란 확인 버튼 터치 -> 1단계 복귀 ({cx}, {cy})")
                        current_step = 1
                        step_start_time = time.time()
                        found = True
                if not found:
                    print(f"[{time.strftime('%H:%M:%S')}] [3단계] 스캔 중 (대기 시간: {int(time.time() - step_start_time)}초)...", end="\r")

        time.sleep(0.5)

except KeyboardInterrupt:
    set_screen_awake(False)
    print("\n\n매크로가 종료되었습니다.")