const canvas = document.getElementById("renderCanvas");
const engine = new BABYLON.Engine(canvas, true);

const numCubes = 20;
const [scene, ground, cubes] = createScene(engine, canvas, numCubes);

const inputMgr = new InputManager(scene)

const avatar = createAvatar(scene, inputMgr);

const isHttps = window.location.protocol === "https:";
// If the page is served over HTTPS, use wss (WebSocket Secure). Otherwise, use ws.
const wsProtocol = isHttps ? "wss://" : "ws://";
const serverAddress = `${wsProtocol}192.168.4.26:8888`;

let ws = null; // WebSocket instance
let reconnectIntervalId = null; // Interval ID for reconnection attempts

let xrHelper;

function assert(condition, message) {
  if (!condition) {
      throw new Error(message || "Assertion failed");
  }
}

function setConnectionState(connected) {
  elem = document.getElementById("connectionState");
  if (connected) {
    elem.textContent = "yes";
    elem.style.color = "green"
  } else {
    elem.textContent = "no";
    elem.style.color = "gray"
  }
}

setConnectionState(false);

function connectWebSocket() {
  // WebSocket setup
  ws = new WebSocket(serverAddress);

  // Event handler for when the connection is established
  ws.onopen = function () {
    console.log("ws.onopen");
    ws.send("vrcube client ready");
    console.log("sent message: vrcube client ready");

    setConnectionState(true);
  };


  // Event handler for errors
  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    setConnectionState(false);
    // Close the WebSocket connection if any error occurs
    ws.close();
  };

  ws.onmessage = function (event) {
    const keyframes = JSON.parse(event.data);

    if (keyframes.length == 0) {
      throw new Error("received empty keyframes array");
    }
    latestKeyframe = keyframes[keyframes.length - 1]

    const keyframeIdx = latestKeyframe.keyframe_index;
    document.getElementById("keyframeIndex").textContent = keyframeIdx;
    // todo: throw error if keyframes came out of order (and beware reconnection)
    const cubePosesData = latestKeyframe.cube_poses

    if (cubePosesData.length != cubes.length) {
      throw new Error(`cubePosesData.length (${cubePosesData.length}) != cubes.length (${cubes.length})`);
    }

    for (let i = 0; i < cubes.length; i++) {
      const cubeData = cubePosesData[i];
      const cube = cubes[i];

      // Update cube pose (rotation and translation)
      cube.rotationQuaternion = new BABYLON.Quaternion(
        cubeData.rotation.x,
        cubeData.rotation.y,
        cubeData.rotation.z,
        cubeData.rotation.w
      );
      cube.position = new BABYLON.Vector3(
        cubeData.translation.x,
        cubeData.translation.y,
        cubeData.translation.z
      );
    }

    receivedCount++;
  };
}

// Initial connection
connectWebSocket();

    // let xrHelper;
    // BABYLON.WebXRDefaultExperience.CreateAsync(scene).then((xr) => {
    //     xrHelper = xr;
    // });


// Button click event handler to enable VR
const vrButton = document.getElementById("vrButton");
vrButton.addEventListener("click", function () {
  try {
    const xrPromise = scene.createDefaultXRExperienceAsync({
      floorMeshes: [ground],
    });

    xrPromise.then((xrExperience) => {
      xrHelper = xrExperience

      if (!xrHelper.baseExperience) {
        console.log("defaultXRExperience.baseExperience is null, indicating no xr support.");
      } else {
        console.log("Done, WebXR is enabled.");
      }

      // todo: clean these up when exiting VR
      inputMgr.addVRListeners(xrHelper);

      engine.runRenderLoop(() => scene.render());
    });
  } catch (error) {
    document.getElementById("errorLog").textContent = error;
  }
});

// Attempt reconnection every X seconds
reconnectIntervalId = setInterval(() => {
  attemptReconnection();
}, 5000);


engine.runRenderLoop(() => {
  scene.render();
});

function getPoseForSerialize(position, rotationQuaternion) {

  const poseObj = {
    position: [position.x, position.y, position.z],
    rotation: [rotationQuaternion.x, rotationQuaternion.y, rotationQuaternion.z, rotationQuaternion.w]
  }
  return poseObj;
}

function getWorldPoseFromBabylonObjectForSerialize(obj) {

  const worldPosition = obj.absolutePosition;
  const worldMatrix = obj.getWorldMatrix();
  const worldRotationQuaternion = BABYLON.Quaternion.FromRotationMatrix(worldMatrix);
  return getPoseForSerialize(worldPosition, worldRotationQuaternion);
}

function doClientUpdate() {

  clientUpdateRate = 60.0;
  const deltaTime = 1.0 / clientUpdateRate;

  if (avatar.handleInputEvents) {
    avatar.handleInputEvents();
  }

  // Note that sending client state and calling inputMgr.onEndFrame are currently
  // tightly coupled; there's a notion of an "input frame", where possibly multiple
  // raw key-down events get treated as a single frame key-down event and passed
  // to the server as a single key-down event (similar for key-up).
  sendClientState();

  inputMgr.onEndFrame();

  setTimeout(doClientUpdate, deltaTime * 1000);
}

function sendClientState() {

  if (avatar && ws && ws.readyState == WebSocket.OPEN) {

    const clientState = {};
    clientState["avatar"] = {
      root: getWorldPoseFromBabylonObjectForSerialize(avatar.root),
      hands: avatar.hands.map(obj => getWorldPoseFromBabylonObjectForSerialize(obj))
    };
    clientState["input"] = inputMgr.getSerializeState()
    
    ws.send(JSON.stringify(clientState));
  }
}

function attemptReconnection() {
  if (!ws || ws.readyState === WebSocket.CLOSED) {
    setConnectionState(false);
    connectWebSocket();
  }
}

doClientUpdate()

// Register OnEveryFrameTrigger to continuously move the box
scene.actionManager.registerAction(new BABYLON.ExecuteCodeAction(
  BABYLON.ActionManager.OnEveryFrameTrigger,
  function () {

    var deltaTime = scene.getEngine().getDeltaTime() / 1000;

    avatar.update(deltaTime);
  }
));

