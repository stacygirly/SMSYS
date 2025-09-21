// document.addEventListener('DOMContentLoaded', function () {
//   const startButton = document.getElementById('startButton');
//   const stopButton = document.getElementById('stopButton');
//   const intervalInput = document.getElementById('intervalInput');

//   startButton.addEventListener('click', () => {
//       chrome.runtime.sendMessage({ action: 'startRecording' }, response => {
//           console.log(response.status);
//           startButton.disabled = true;
//           stopButton.disabled = false;
//       });
//   });

//   stopButton.addEventListener('click', () => {
//       chrome.runtime.sendMessage({ action: 'stopRecording' }, response => {
//           console.log(response.status);
//           startButton.disabled = false;
//           stopButton.disabled = true;
//       });
//   });

//   // Check the recording status when the popup is opened
//   chrome.runtime.sendMessage({ action: 'getStatus' }, (response) => {
//       if (response.isRecording) {
//           startButton.disabled = true;
//           stopButton.disabled = false;
//       } else {
//           startButton.disabled = false;
//           stopButton.disabled = true;
//       }
//   });
// });
