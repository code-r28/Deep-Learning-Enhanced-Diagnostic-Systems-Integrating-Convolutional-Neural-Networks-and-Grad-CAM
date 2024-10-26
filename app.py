from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import cv2
from fpdf import FPDF
from PIL import Image
import time
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


model = load_model('monkeypox_model_t.h5')


progress = 0

def update_progress(value):
    global progress
    progress = value


def convert_webp_to_jpg(img_path):
    if img_path.endswith('.webp'):
        img = Image.open(img_path)
        img_path_jpg = img_path.replace('.webp', '.jpg')
        img = img.convert('RGB') 
        img.save(img_path_jpg, 'JPEG')
        return img_path_jpg
    return img_path


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="uploads/cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))


    heatmap = np.uint8(255 * heatmap)


    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img


    cv2.imwrite(cam_path, superimposed_img)

    return cam_path


def generate_report(username, img_path, predicted_class, confidence, heatmap_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 16)
    centered_image_path = 'static/images/new-sriher-logo.png'  
    image_width = 60  
    page_width = 210  
    x_position = (page_width - image_width) / 2  
    pdf.image(centered_image_path, x=x_position, y=8, w=image_width) 


    pdf.ln(10)  
    pdf.cell(190, 10, 'Monkeypox Diagnostic Report', 0, 1, 'C')


    pdf.set_font('Arial', 'I', 10)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(190, 10, f'Report Generated on: {current_time}', 0, 1, 'C')
    

    pdf.set_line_width(0.5)
    pdf.rect(5, 5, 200, 287) 


    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, f'Dear {username}, based on the analysis:')


    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(40, 10, f'Predicted Class: {predicted_class}')
    pdf.ln(10)
    pdf.cell(40, 10, f'Confidence Score: {confidence:.2f}%')
    pdf.ln(10)


    img_path = convert_webp_to_jpg(img_path)
    heatmap_path = convert_webp_to_jpg(heatmap_path)
    pdf.image(img_path, x=10, y=60, w=90)  
    pdf.image(heatmap_path, x=110, y=60, w=90)  


    pdf.ln(90)
    pdf.cell(90, 10, 'Your Skin', 0, 0, 'C')
    pdf.cell(90, 10, 'Focused Area (Heatmap)', 0, 1, 'C')
    
    pdf.ln(10)  

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(40, 10, 'Explanation of Heatmap')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)

    if predicted_class == 'Monkeypox':
        pdf.multi_cell(0, 10, "The heatmap highlights specific areas of the skin lesion that the model identified as "
                              "critical in diagnosing Monkeypox. The red and yellow regions indicate the parts of the image "
                              "that were most important in the model's decision. These areas may correspond to characteristic "
                              "Monkeypox features such as blistering, ulceration, or specific lesion patterns.")
    else:
        pdf.multi_cell(0, 10, "The heatmap highlights areas of the skin lesion that the model identified. However, based "
                              "on the prediction, the highlighted regions do not show patterns typically associated with "
                              "Monkeypox lesions. The absence of strong activation in the image suggests a negative prediction "
                              "for Monkeypox.")


    pdf.set_font('Arial', 'B', 14)
    pdf.cell(40, 10, 'Next Steps')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)


    if predicted_class == 'Monkeypox':
        pdf.multi_cell(0, 10, "Based on the prediction, it is advised to consult with a healthcare professional immediately for further testing and confirmation. "
                              "Until you receive professional advice, try to limit contact with others to avoid potential transmission of the virus. "
                              "Monitor symptoms like fever, swollen lymph nodes, or additional skin lesions. Testing for Monkeypox-specific antigens may be required.")
    else:
        pdf.multi_cell(0, 10, "The result suggests that the lesion is not Monkeypox. However, if symptoms worsen or additional lesions appear, it is advisable to "
                              "re-consult a healthcare professional. Keep monitoring the symptoms and follow general skin care practices to maintain hygiene.")


    timestamp = int(time.time())
    report_path = f'static/reports/diagnostic_report_{timestamp}.pdf'
    pdf.output(report_path)
    return report_path


def send_email_with_report(username, confidence, report_path):
    sender_email = '@gmail.com'  #Sender mail address here
    sender_password = ''  #Sender mail address pwd here generated from Google
    receiver_email = '' #Receiver mail address here  

    subject = f"Monkeypox Detection Alert for {username}"
    body = f"""\
Dear Medical Professional,

Patient {username} has been detected with Monkeypox with a confidence of {confidence:.2f}%.
Please find the attached report for more details.

Best regards,
Monkeypox Diagnostic System
"""


    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject


    message.attach(MIMEText(body, 'plain'))


    with open(report_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {os.path.basename(report_path)}',
        )
        message.attach(part)


    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/progress', methods=['GET'])
def progress_update():
    global progress
    return jsonify({'progress': progress})


@app.route('/predict', methods=['POST'])
def predict():
    global progress
    progress = 10  
    
    username = request.form['username']  
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
       
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        progress = 20  


        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        progress = 40 


        prediction = model.predict(img_array)
        predicted_class = 'Monkeypox' if np.argmax(prediction) == 0 else 'Non-Monkeypox'
        confidence = np.max(prediction) * 100

        progress = 60  


        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block16_0_bn")
        heatmap_path = save_and_display_gradcam(filepath, heatmap, cam_path=os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.jpg'))

        progress = 80  


        report_path = generate_report(username, filepath, predicted_class, confidence, heatmap_path)

        progress = 100  


        if confidence > 95 and predicted_class == 'Monkeypox':
            send_email_with_report(username, confidence, report_path)

        return render_template('result.html', username=username, prediction=predicted_class, confidence=confidence, image_path=filepath, heatmap_path=heatmap_path, report_path=report_path)

if __name__ == '__main__':
    app.run(debug=True)
