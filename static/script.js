const userInput = document.getElementById('user-input');
const suggestion = document.getElementById('suggestion');
const chatHistory = document.getElementById('chat-history');
const modelInfo = document.getElementById('model-info');
const stats = document.getElementById('stats');
const sendButton = document.getElementById('send-button');

let slowModel = false;
let debounceTimer;
let currentSuggestion = null;

let typedKeystrokes = 0;
let acceptedSuggestionChars = 0;
let totalSessionKeystrokes = 0;
let totalSessionAcceptedChars = 0;

let lastEventId = null;
let eventCounter = 0;

function generateEventId() {
    eventCounter++;
    return `event-${eventCounter}`;
}

function updateStats() {
    const totalChars = totalSessionKeystrokes + totalSessionAcceptedChars + typedKeystrokes + acceptedSuggestionChars;
    const effortSaved = totalSessionAcceptedChars + acceptedSuggestionChars;
    const effortSavedPercentage = totalChars > 0 ? (effortSaved / totalChars * 100).toFixed(2) : 0;
    stats.innerHTML = `Total Effort Saved: ${effortSavedPercentage}%`;
}

async function logEvent(eventType, details, parentEventId) {
    const eventId = generateEventId();
    
    // Safeguard against parentEventId being an object
    const cleanParentEventId = (parentEventId && typeof parentEventId === 'object') ? null : parentEventId;

    await fetch('/log_event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            event_type: eventType,
            details: details,
            event_id: eventId,
            parent_event_id: cleanParentEventId
        })
    });
    lastEventId = eventId;
    return eventId;
}

async function logAndClearSuggestion(status, details, parentEventId) {
    if (currentSuggestion) {
        if (status === 'suggestion_accepted') {
            acceptedSuggestionChars += currentSuggestion.length - userInput.value.length;
        }
        const eventId = await logEvent(status, { suggestion: currentSuggestion, ...details }, parentEventId);
        currentSuggestion = null;
        suggestion.textContent = '';
        return eventId;
    }
    return parentEventId;
}

async function loadChatHistory() {
    const response = await fetch('/get_context');
    const conversation = await response.json();
    chatHistory.innerHTML = '';
    conversation.forEach(utterance => {
        const p = document.createElement('p');
        
        // Create a div to hold both image and text
        const messageDiv = document.createElement('div');
        messageDiv.style.marginBottom = '15px';
        
        // Check if this utterance has an image
        if (utterance.image_path) {
            const img = document.createElement('img');
            img.src = utterance.image_path;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '300px';
            img.style.display = 'block';
            img.style.marginBottom = '10px';
            messageDiv.appendChild(img);
        }
        
        // Add the text message
        p.innerHTML = `<strong>${utterance.speaker}:</strong> ${utterance.message || utterance.utterance}`;
        messageDiv.appendChild(p);
        
        // Add the complete message div to chat history
        chatHistory.appendChild(messageDiv);
    });
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function setModel() {
    const response = await fetch('/set_model', { method: 'POST' });
    const data = await response.json();
    slowModel = data.slow_model;
    modelInfo.innerHTML = `Using ${data.model_name}. ${slowModel ? '<strong>This is a slow model, suggestions will appear on pause.</strong>' : ''}`;
    await logEvent('model_assigned', { model_name: data.model_name, slow_model: data.slow_model }, lastEventId);
}

async function sendMessage(parentEventId) {
    const text = userInput.value.trim();
    if (text) {
        const eventId = await logAndClearSuggestion('suggestion_rejected', { final_text: text }, parentEventId);
        const p = document.createElement('p');
        p.innerHTML = `<strong>You:</strong> ${text}`;
        chatHistory.appendChild(p);
        
        totalSessionKeystrokes += typedKeystrokes;
        totalSessionAcceptedChars += acceptedSuggestionChars;
        typedKeystrokes = 0;
        acceptedSuggestionChars = 0;

        userInput.value = '';
        chatHistory.scrollTop = chatHistory.scrollHeight;
        updateStats();
        
        // No longer showing rating popup here since it's already shown before calling sendMessage
        // Just return the event ID
        return eventId;
    }
    return parentEventId;
}

async function getSuggestion(parentEventId) {
    const text = userInput.value;
    if (!text) {
        await logAndClearSuggestion('suggestion_rejected', { final_text: text }, parentEventId);
        return;
    }

    // Get all paragraphs and images in the chat history
    const historyElements = chatHistory.querySelectorAll('p');
    const imageElements = chatHistory.querySelectorAll('img');
    
    // Create a map to associate images with their corresponding utterances
    const imageMap = new Map();
    let currentIndex = 0;
    
    // Process all elements in order
    const allElements = Array.from(chatHistory.childNodes);
    let currentSpeaker = null;
    
    const chat_history = [];
    
    for (let i = 0; i < allElements.length; i++) {
        const element = allElements[i];
        
        // If it's an image, store it to associate with the next paragraph
        if (element.tagName === 'IMG') {
            const imgSrc = element.src;
            currentIndex = chat_history.length;
            imageMap.set(currentIndex, imgSrc);
        }
        // If it's a paragraph, extract the speaker and message
        else if (element.tagName === 'P') {
            const speakerElement = element.querySelector('strong');
            if (speakerElement) {
                const speaker = speakerElement.textContent.replace(':', '');
                const message = element.textContent.replace(speakerElement.textContent, '').trim();
                
                // Create the chat history item
                const historyItem = { speaker, message };
                
                // If there's an image associated with this utterance, add it
                if (imageMap.has(chat_history.length)) {
                    historyItem.image_path = imageMap.get(chat_history.length);
                }
                
                chat_history.push(historyItem);
            }
        }
    }

    const response = await fetch('/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, chat_history })
    });
    const data = await response.json();
    
    if (data.suggestion) {
        suggestion.textContent = data.suggestion;
        currentSuggestion = data.suggestion;
        await logEvent('suggestion_provided', { prefix: text, suggestion: data.suggestion }, parentEventId);
    } else {
        await logAndClearSuggestion('suggestion_rejected', { final_text: text }, parentEventId);
    }
}

const ratingPopup = document.getElementById('rating-popup');
const ratingOptions = document.getElementById('rating-options');
let ratingResolve = null;

function showRatingPopup() {
    ratingPopup.classList.remove('hidden');
    ratingOptions.innerHTML = '';
    const prompt = document.createElement('p');
    prompt.textContent = 'type on the keyboard';
    ratingOptions.appendChild(prompt);
    for (let i = 0; i < 10; i++) {
        const span = document.createElement('span');
        span.textContent = i;
        ratingOptions.appendChild(span);
    }
    return new Promise(resolve => {
        ratingResolve = resolve;
        document.addEventListener('keydown', handleRatingKeydown);
    });
}

function hideRatingPopup() {
    ratingPopup.classList.add('hidden');
    document.removeEventListener('keydown', handleRatingKeydown);
}

function handleRatingKeydown(e) {
    if (e.key >= '0' && e.key <= '9') {
        e.preventDefault();
        if (ratingResolve) {
            ratingResolve(parseInt(e.key, 10));
            ratingResolve = null;
        }
        hideRatingPopup();
    }
}

userInput.addEventListener('input', async () => {
    userInput.value = userInput.value.toLowerCase();
    const text = userInput.value;
    let parentEventId = lastEventId;
    if (currentSuggestion && !currentSuggestion.startsWith(text)) {
        parentEventId = await logAndClearSuggestion('suggestion_rejected', { final_text: text }, parentEventId);
    }

    if (slowModel) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => getSuggestion(parentEventId), 500);
    } else {
        getSuggestion(parentEventId);
    }
});

userInput.addEventListener('keydown', async (e) => {
    if (document.activeElement !== userInput) {
        return;
    }
    let parentEventId = lastEventId;
    if (e.key.length === 1) {
        typedKeystrokes++;
    } else if (e.key === 'Backspace') {
        // This is a simplification. It doesn't handle deleting selected text.
        if (userInput.value.length > 0) {
            typedKeystrokes = Math.max(0, typedKeystrokes - 1);
        }
    }
    
    if (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Enter') {
        parentEventId = await logEvent('keystroke', { key: e.key, prefix: userInput.value }, parentEventId);
    }

    if (currentSuggestion && (e.key === 'Tab' || e.key === 'ArrowRight')) {
        e.preventDefault();
        const prefix = userInput.value;
        const originalLength = userInput.value.length;
        // Make sure we're appending the suggestion to the prefix, not replacing it
        userInput.value = prefix + currentSuggestion.substring(prefix.length);
        acceptedSuggestionChars += currentSuggestion.length - originalLength;
        await logAndClearSuggestion('suggestion_accepted', {suggestion: currentSuggestion, prefix: prefix}, parentEventId);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        const rating = await showRatingPopup();
        parentEventId = await logEvent('completed', { final_text: userInput.value, rating }, parentEventId);
        await sendMessage(parentEventId);
        // Reload with a new example
        location.href = '/?new_example=true'
    }
    updateStats();
});

async function initializeApp() {
    await loadChatHistory();
    await setModel();
    updateStats();
}

initializeApp();

sendButton.addEventListener('click', async () => {
    // Show rating popup first, just like the Enter key handler
    const rating = await showRatingPopup();
    parentEventId = await logEvent('completed', { final_text: userInput.value, rating }, parentEventId);
    await sendMessage(parentEventId);
    // Reload with a new example
    location.href = '/?new_example=true';
});