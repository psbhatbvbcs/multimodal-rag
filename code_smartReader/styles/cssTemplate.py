css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: rgba(0,0,0,0.07);
    border: 1px solid rgba(0, 0, 0, 0.16);
    text-align: left;
    flex-direction: row;
    margin-right: 5rem;
}
.chat-message.bot {
    background-color: rgba(0,0,899,0.07);
    border: 1px solid rgba(0, 0, 0, 0.16);
    text-align: left;
    justify-content: flex-end;
    margin-left: 5rem;
}
.chat-message .avatar {
    width: 20%;
    display: flex;
    justify-content: center;
}
.chat-message .avatar img {
    max-width: 50px;
    max-height: 50px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #000;
}

.conversation-container {
    max-height: 400px; /* Adjust the height as needed */
    overflow-y: auto;
}
</style>
'''
