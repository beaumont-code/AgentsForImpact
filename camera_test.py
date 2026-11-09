import cv2


def main() -> None:
    # On macOS, CAP_AVFOUNDATION is usually the most reliable backend.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0).")
        print("Tip: Check System Settings > Privacy & Security > Camera permissions.")
        return

    print("Webcam started. Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
