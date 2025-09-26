// chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
//     if (request.action === 'startRecording') {
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//             const mediaRecorder = new MediaRecorder(stream);

//             mediaRecorder.ondataavailable = (event) => {
//                 if (event.data.size > 0) {
//                     const blob = new Blob([event.data], { type: 'audio/wav' });
//                     chrome.runtime.sendMessage({ action: 'audioBlob', blob });
//                 }
//             };

//             mediaRecorder.start();
//             console.log('Recording started');
//             sendResponse({ status: 'recording started' });
//         } catch (error) {
//             console.error('Error accessing microphone:', error);
//             sendResponse({ status: 'failed to start recording' });
//         }
//     } else if (request.action === 'stopRecording') {
//         // Logic for stopping the recording
//         sendResponse({ status: 'recording stopped' });
//     }
// });
