let currentEntryIndex = 0;
let entries = [];

document.addEventListener('keydown', handleKeydown);
document.getElementById('file-button').addEventListener('click', function() {
    document.getElementById('file-input').click();
});
document.getElementById('file-input').addEventListener('change', loadFile);

function loadFile(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('file-name').innerText = file.name;
        const reader = new FileReader();
        reader.onload = function(e) {
            entries = JSON.parse(e.target.result);
            currentEntryIndex = 0;
            updateEntry();
        };
        reader.readAsText(file);
    }
}

function updateEntry() {
    if (entries.length === 0) {
        document.getElementById('question').innerText = "No entries available.";
        return;
    }

    const entry = entries[currentEntryIndex];
    document.getElementById('question-counter').innerText = `Question: ${currentEntryIndex + 1}/${entries.length}`;
    document.getElementById('question').innerText = entry.item.question;

    if (entry.correct_letter === 'a') {
        setArgumentDetails('a', entry.item.answer_correct, entry.debater_a, entry.response_a, true);
        setArgumentDetails('b', entry.item.answer_incorrect, entry.debater_b, entry.response_b, false);
    } else {
        setArgumentDetails('a', entry.item.answer_incorrect, entry.debater_a, entry.response_a, false);
        setArgumentDetails('b', entry.item.answer_correct, entry.debater_b, entry.response_b, true);
    }

    document.getElementById('judge').innerText = entry.judge;
    document.getElementById('judge-confidence').innerText = entry.judge_confidence.toFixed(2);
    document.getElementById('naive-judge-confidence').innerText = entry.naive_judge_confidence.toFixed(2);

    updateConfidenceBar('judge-bar', entry.judge_confidence);
    updateConfidenceBar('naive-judge-bar', entry.naive_judge_confidence);
}

function setArgumentDetails(column, answerDetails, debater, response, isCorrect) {
    document.getElementById(`proof-${column}`).innerText = answerDetails.proof;
    document.getElementById(`numeric-${column}`).innerText = answerDetails.numeric;
    document.getElementById(`debater-${column}`).innerText = debater;
    document.getElementById(`response-${column}`).innerHTML = DOMPurify.sanitize(marked.parse(response));

    const argumentElement = document.getElementById(`argument-${column}`);
    argumentElement.classList.remove('correct', 'incorrect');
    if (isCorrect) {
        argumentElement.classList.add('correct');
    } else {
        argumentElement.classList.add('incorrect');
    }
}

function updateConfidenceBar(barId, confidenceValue) {
    const barElement = document.getElementById(barId);
    barElement.style.width = 100.0 * confidenceValue + '%';
}

function nextEntry() {
    if (currentEntryIndex < entries.length - 1) {
        currentEntryIndex++;
        updateEntry();
    }
}

function previousEntry() {
    if (currentEntryIndex > 0) {
        currentEntryIndex--;
        updateEntry();
    }
}

function handleKeydown(event) {
    if (event.key === 'ArrowRight') {
        nextEntry();
    } else if (event.key === 'ArrowLeft') {
        previousEntry();
    }
}
