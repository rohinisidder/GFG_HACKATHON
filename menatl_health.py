import numpy as np
import pandas as pd

# Simulate data for 30 days
np.random.seed(42)
days = 30
students = 50

# Simulate screen time data in minutes
screen_time_normal = np.random.normal(loc=120, scale=30, size=(students, days))
# Adding some anomalous data
screen_time_anomalous = screen_time_normal.copy()
anomalous_indices = np.random.choice(range(students), size=int(0.1 * students), replace=False)
for idx in anomalous_indices:
    screen_time_anomalous[idx, np.random.choice(range(days), size=3, replace=False)] *= 2

# Simulate website visits
websites = ['social_media', 'educational', 'self_harm', 'gaming']
website_visits_normal = np.random.choice(websites[:-1], size=(students, days))
website_visits_anomalous = website_visits_normal.copy()
for idx in anomalous_indices:
    website_visits_anomalous[idx, np.random.choice(range(days), size=3, replace=False)] = 'self_harm'

# Create DataFrame
data_normal = pd.DataFrame(screen_time_normal, columns=[f'day_{i+1}' for i in range(days)])
data_anomalous = pd.DataFrame(screen_time_anomalous, columns=[f'day_{i+1}' for i in range(days)])
data_normal['student_id'] = range(students)
data_anomalous['student_id'] = range(students)
data_normal['website_visits'] = website_visits_normal.tolist()
data_anomalous['website_visits'] = website_visits_anomalous.tolist()

# Merge data for simulation
data = data_anomalous.copy()
data['normal_screen_time'] = data_normal.drop('student_id', axis=1).values.tolist()
data['normal_website_visits'] = data_normal['website_visits']

# Save to CSV for later use
data.to_csv('simulated_student_data.csv', index=False)
















import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the data
file_path = 'simulated_student_data.csv'
data = pd.read_csv(file_path)

# Convert lists back from string representation
data['website_visits'] = data['website_visits'].apply(eval)
data['normal_screen_time'] = data['normal_screen_time'].apply(eval)
data['normal_website_visits'] = data['normal_website_visits'].apply(eval)

# Extract features
features = []
for i in range(data.shape[0]):
    row = data.iloc[i]
    normal_screen_time = np.array(row['normal_screen_time'][:days], dtype=float)
    anomalous_screen_time = np.array([float(val) for val in row[[f'day_{i+1}' for i in range(days)]]])
    
    features.append([
        np.mean(normal_screen_time),
        np.std(normal_screen_time),
        np.mean(anomalous_screen_time),
        np.std(anomalous_screen_time),
        row['website_visits'].count('self_harm'),
        row['normal_website_visits'].count('self_harm')
    ])

features = pd.DataFrame(features, columns=[
    'mean_normal_screen_time', 'std_normal_screen_time',
    'mean_anomalous_screen_time', 'std_anomalous_screen_time',
    'self_harm_visits_anomalous', 'self_harm_visits_normal'
])

# Labels for anomaly detection
labels = (features['self_harm_visits_anomalous'] > 0).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))




import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv, dotenv_values


load_dotenv()  # This loads the .env file into the environment

# Access environment variables directly from the dotenv module
config = dotenv_values(".env")
# Email configuration
SMTP_SERVER = 'smtp.gmail.com'  # SMTP server address
SMTP_PORT = 587  # SMTP port
EMAIL_ADDRESS = config.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = config.get('EMAIL_PASSWORD')

# Function to send email notification
def send_notification(student_id, recipient_email):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg['Subject'] = 'Anomaly Detected in Student Behavior'
    
    body = f"Notification: Anomaly detected for student {student_id}. Parents have been notified."
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    text = msg.as_string()
    server.sendmail(EMAIL_ADDRESS, recipient_email, text)
    server.quit()

# Simulate new data
new_data = data.sample(5)
new_features = []
for i in range(new_data.shape[0]):
    row = new_data.iloc[i]
    normal_screen_time = np.array(row['normal_screen_time'][:days], dtype=float)
    anomalous_screen_time = np.array([float(val) for val in row[[f'day_{i+1}' for i in range(days)]]])
    
    new_features.append([
        np.mean(normal_screen_time),
        np.std(normal_screen_time),
        np.mean(anomalous_screen_time),
        np.std(anomalous_screen_time),
        row['website_visits'].count('self_harm'),
        row['normal_website_visits'].count('self_harm')
    ])

new_features = pd.DataFrame(new_features, columns=[
    'mean_normal_screen_time', 'std_normal_screen_time',
    'mean_anomalous_screen_time', 'std_anomalous_screen_time',
    'self_harm_visits_anomalous', 'self_harm_visits_normal'
])

# Detect anomalies
anomalies = model.predict(new_features)
new_data['anomaly_detected'] = anomalies

print(new_data[['student_id', 'anomaly_detected']])

# Simulate notifications
for idx, row in new_data.iterrows():
    if row['anomaly_detected']:
        # You need to have the email addresses of the parents/recipients in your dataset or predefined
        recipient_email = 'recipient_email@example.com'  # Replace with the actual recipient email
        send_notification(row['student_id'], recipient_email)
