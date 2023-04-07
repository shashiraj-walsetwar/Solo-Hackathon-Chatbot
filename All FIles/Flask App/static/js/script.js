
// Get the chat form, input field, and chat messages container
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('user_message');
const chatMessages = document.getElementById('chat-messages');

// Handle form submission
chatForm.addEventListener('submit', (event) => {
  event.preventDefault();

  // Get user's message
  const message = chatInput.value;
  // console.log("Got User Message: " + message)

  // Display user's message in the chat
  displayMessage('User', message);

  // Ajax Request for sending message to get-response endpoint

  $.ajax({
    data: {
        msg: message,
    },
    type: "POST",
    url: "/get-response",
}).done(function(data) {
    displayMessage("Bot", data)
});

  // Clear input field
  chatInput.value = '';
});

// Display message in the chat
function displayMessage(sender, message) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('chat-message');
  messageElement.innerHTML = `
    <span class="sender">${sender}:</span>
    <span class="message-text">${message}</span>
  `;
  chatMessages.appendChild(messageElement);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
