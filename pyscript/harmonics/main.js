// see https://web.dev/media-recording-audio/#access_the_raw_data_from_the_microphone 
// see https://github.com/Honghe/demo_fastapi_websocket/blob/master/README.md#access-the-raw-data-from-the-microphone 

const player = document.getElementById('player');
const refreshPeriod = 1500.0; // in milliseconds
const bufferLength = 1024//16384
;
var lastTime = 0;

const processFcn = function (evt) {
// AUDIO PROCESSING FUNCTION
    if ((performance.now()-lastTime)>refreshPeriod) {
        lastTime = performance.now();
        try {
            let processFcn_py = pyscript.interpreter.globals.get('processFcn');
            // console.log(evt);
            processFcn_py(evt);
            console.log("processed audio in " + (performance.now()-lastTime) + "ms");
            return;
        } 
        catch{}
    }
    console.log("no audio process..");
    };

const initProcessor = function(stream){

    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
  
    // USING AUDIO WORKLET (NEEDS EXTERNAL JS FILE FOR PROCESSOR..)
    // await context.audioWorklet.addModule("./processor.js");
    // const worklet = new AudioWorkletNode(context, "worklet-processor");
    // source.connect(worklet);
    // worklet.connect(context.destination);
    
    // USING DEPRECATED SCRIPT PROCESSOR
    const processor = context.createScriptProcessor(bufferLength, 1, 1);
    processor.onaudioprocess = processFcn
    source.connect(processor);
    processor.connect(context.destination);

}
const handleSuccess = async function (stream) {
  if (window.URL) {
    player.srcObject = stream;
  } else {
    player.src = stream;
  }
  initProcessor(stream);
};


// const getUserMedia = require('get-user-media-promise');

// function getConnectedDevices() {
//     navigator.mediaDevices.enumerateDevices()
//         .then(devices => {
//             console.log(devices);
//         });
// }
// getConnectedDevices();


navigator.mediaDevices
  .getUserMedia({audio: true, video: false})
  .then(handleSuccess);