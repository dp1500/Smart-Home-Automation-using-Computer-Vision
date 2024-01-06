from flask import Flask, request
import firebase_admin
from firebase_admin import credentials, messaging
from firebase_admin import messaging

# Initialize Firebase app (replace with your Firebase configuration)
app = Flask(__name__)
cred = credentials.Certificate('smart_home_automation_2\home-automation-f4916-firebase-adminsdk-eu45p-bf29f14315.json')
firebase_admin.initialize_app(cred)

@app.route('/')
def index():
    return 'Notification test app'

@app.route('/send_notification', methods=['POST'])
def send_notification():
    # token = "cru-1SEKSHi5R6wPU0n6Lt:APA91bHx9yrK9P_t_yLSQS5tvnn95Z_3XIEkY-5QIbnbkx4Giz9GaBhEODrhNgxt9Djm4VmTCD5ue5N08pfWao2vm3MgVFk3JkFGDP3uiH1oPthYQeIb7oNOgpNXo76MyfvzvyfU9m2w"
    title = request.args.get('title')
    body = request.args.get('body')

    # Create the notification payload with both notification and data fields
    push_message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        data={
            # Additional data for your app
        },
        # token=token
        topic = "notify"
    )

    try:
        response = messaging.send(push_message)
        print('Notification sent:', response)
        return 'Notification sent successfully'
    except Exception as e:
        print('Failed to send notification:', e)
        return 'Notification failed to send', 500

if __name__ == '__main__':
    app.run(debug=True)
