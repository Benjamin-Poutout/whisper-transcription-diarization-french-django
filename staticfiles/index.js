let currentAudioFile = null;  // Fichier audio initial ou concaténé
let audioChunks = [];
let mediaRecorder;
let recordingInterval;
const chunkDuration = 3000;  // 3 secondes
let isRecording = false;  // Flag pour vérifier si l'enregistrement est en cours

// Connexion WebSocket
const socket = new WebSocket('ws://127.0.0.1:8000/ws/transcription/');

socket.onopen = () => {
    console.log("Connected to WebSocket");
};

// Envoyer le nombre de locuteurs dès que l'utilisateur clique sur le bouton
const setSpeakersButton = document.getElementById("setSpeakersButton");
if (setSpeakersButton) {
    setSpeakersButton.addEventListener("click", () => {
        const numSpeakers = parseInt(document.getElementById("numSpeakers").value, 10);
        if (!isNaN(numSpeakers)) {
            socket.send(JSON.stringify({ type: 'set_speakers', numSpeakers: numSpeakers }));
            console.log(`Number of speakers set to: ${numSpeakers}`);
        } else {
            console.error("Invalid number of speakers");
        }
    });
} else {
    console.error("setSpeakersButton not found");
}

socket.onmessage = (event) => {
    // Convertir l'événement en JSON
    const message = JSON.parse(event.data);

    // Vérifier le type du message et afficher les informations correspondantes
    if (message.type === "transcription") {
        const transcription = message.data;
        console.log("Transcription received: ", transcription);
        document.getElementById("transcription").innerText = transcription;  // Afficher la transcription
    } else if (message.type === "diarization") {
        const diarization = message.data;
        console.log("Diarization received: ", diarization);
        
        // Afficher la diarisation dans la zone prévue
        document.getElementById("diarization").innerText = diarization;

        // Rendre la section de résultat visible
        const resultSection = document.getElementById("result-section");
        resultSection.style.display = "block";  // Rendre la section visible

        // Vous pouvez également ajouter une option pour télécharger le fichier (format choisi)
        // Exemple : générer un fichier JSON ou CSV à partir des résultats de la diarisation
        const saveButton = document.getElementById("save-btn");
        saveButton.addEventListener("click", () => {
            const formatSelect = document.getElementById("format-select");
            const selectedFormat = formatSelect.value;

            let fileContent = '';
            if (selectedFormat === 'json') {
                fileContent = JSON.stringify({ diarization: diarization }, null, 2);
            } else if (selectedFormat === 'csv') {
                fileContent = "speaker,start,end,transcription\n";
                diarization.forEach(item => {
                    fileContent += `${item.speaker},${item.start},${item.end},${item.transcription}\n`;
                });
            } else if (selectedFormat === 'txt') {
                diarization.forEach(item => {
                    fileContent += `${item.speaker}: ${item.transcription}\n`;
                });
            }

            // Créer un Blob avec le contenu du fichier
            const blob = new Blob([fileContent], { type: selectedFormat === 'json' ? 'application/json' : 'text/plain' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `diarization.${selectedFormat}`; // Nom du fichier
            link.click(); // Déclenche le téléchargement
        });
    }
};

socket.onclose = () => {
    console.log("Disconnected from WebSocket");
};

// Fonction qui sera appelée à chaque fois qu'un chunk est capturé
function createWebmChunk(chunkData) {
    const blob = new Blob(chunkData, { type: 'audio/webm' });
    console.log(`Created blob with size: ${blob.size}`);

    if (currentAudioFile) {
        console.log("Concatenating with previous file...");
        concatFiles(currentAudioFile, blob).then((combinedBlob) => {
            console.log("Concatenation complete, sending...");
            socket.send(combinedBlob);

            // Mettre à jour le fichier initial (pour la prochaine concaténation)
            currentAudioFile = combinedBlob;

            // Réinitialiser les morceaux après envoi
            audioChunks = [];
        }).catch(err => {
            console.error("Error during concatenation:", err);
        });
    } else {
        console.log("First chunk, sending directly...");
        socket.send(blob);
        currentAudioFile = blob;  // Sauvegarder le premier chunk comme fichier de base
    }
}

// Fonction pour concaténer les fichiers WebM
async function concatFiles(file1, file2) {
    // Créer des ArrayBuffers à partir des fichiers Blob
    const file1Buffer = await fileToArrayBuffer(file1);
    const file2Buffer = await fileToArrayBuffer(file2);

    // Concaténer les buffers
    const combinedBuffer = new Uint8Array(file1Buffer.byteLength + file2Buffer.byteLength);
    combinedBuffer.set(new Uint8Array(file1Buffer), 0);
    combinedBuffer.set(new Uint8Array(file2Buffer), file1Buffer.byteLength);

    // Créer un nouveau Blob combiné
    return new Blob([combinedBuffer], { type: 'audio/webm' });
}

// Fonction pour convertir un fichier en ArrayBuffer
function fileToArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

// Fonction de gestion de l'enregistrement
const startButton = document.getElementById("startRecording");
const stopButton = document.getElementById("stopRecording");

startButton.onclick = async function () {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    startButton.disabled = true;
    stopButton.disabled = false;

    // Capture audio en chunks
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            console.log(`Received audio chunk of size: ${event.data.size}`);
            audioChunks.push(event.data);
        } else {
            console.log('Received empty audio chunk');
        }
    };

    mediaRecorder.onstop = () => {
        console.log("Recording stopped.");
        if (audioChunks.length > 0) {
            createWebmChunk(audioChunks);
        }
        audioChunks = [];  // Clear the chunks after processing
        isRecording = false;  // L'enregistrement est terminé
    };

    // Start recording
    mediaRecorder.start();
    console.log("Recording started...");
    isRecording = true;  // L'enregistrement est en cours

    // Démarrer un intervalle pour découper en morceaux de 3 secondes
    recordingInterval = setInterval(() => {
        if (isRecording && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();  // Stop the current recording chunk
            console.log("Recording stopped for chunk...");

            // Ajouter un délai avant de redémarrer l'enregistrement
            setTimeout(() => {
                if (mediaRecorder.state !== 'recording') {
                    console.log("Restarting recording...");
                    mediaRecorder.start();  // Redémarrer l'enregistrement pour le prochain chunk
                    isRecording = true;  // Mettre à jour l'état de l'enregistrement
                }
            }, 200);  // 200ms pour permettre à l'enregistreur de se réinitialiser
        }
    }, chunkDuration);  // 3 secondes
};

// Gestion de l'arrêt de l'enregistrement
stopButton.onclick = () => {
    clearInterval(recordingInterval);  // Arrêter l'intervalle d'enregistrement
    mediaRecorder.stop();  // Stopper définitivement l'enregistrement
    startButton.disabled = false;
    stopButton.disabled = true;
    console.log("Recording stopped.");

    // Envoyer un signal au serveur pour lui indiquer que l'enregistrement est terminé
    socket.send(JSON.stringify({ message: 'stop' }));
    console.log("Message 'stop' sent.");

    // Réinitialiser les morceaux pour préparer un nouvel enregistrement
    audioChunks = [];  // Vider les morceaux d'audio
};
