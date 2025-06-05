import cv2

def draw_info_card(frame, name, dob, phone, box, img_path):
    x1, y1, x2, y2 = box
    img_h, img_w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    thickness = 2
    padding = 20
    line_spacing = 40
    face_size = 130

    text_lines = [f"Name: {name}", f"DOB: {dob}", f"Phone: {phone}"]
    text_widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines]
    max_text_width = max(text_widths)

    card_height = max(len(text_lines) * line_spacing + 2 * padding, face_size + 2 * padding)
    card_width = face_size + max_text_width + 4 * padding

    card_x = x2 + 30
    card_y = max(y1 - 10, 10)

    if card_x + card_width > img_w:
        card_x = max(x1 - card_width - 30, 10)

    if card_y + card_height > img_h:
        card_y = max(img_h - card_height - 10, 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + card_height), (255, 255, 255), -1)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.rectangle(frame, (card_x, card_y), (card_x + card_width, card_y + card_height), (0, 0, 0), 2)
    divider_x = card_x + face_size + padding
    cv2.line(frame, (divider_x, card_y), (divider_x, card_y + card_height), (0, 0, 0), 2)

    # Face photo
    try:
        face_img = cv2.imread(img_path)
        if face_img is not None:
            resized_img = cv2.resize(face_img, (face_size, face_size))
            img_y_offset = card_y + (card_height - face_size) // 2
            frame[img_y_offset:img_y_offset + face_size, card_x + padding:card_x + padding + face_size] = resized_img
        else:
            cv2.putText(frame, "Image not found", (card_x + 15, card_y + 60), font, 0.5, (0, 0, 0), 1)
    except Exception as e:
        print("Error loading image:", e)

    text_x = divider_x + padding
    text_y = card_y + padding + 30
    for line in text_lines:
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        text_y += line_spacing
