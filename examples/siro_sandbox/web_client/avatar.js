
// Function to create a user-controlled box in Babylon.js
function createAvatar(scene) {
  // Create a box as the root of the avatar
  var root = BABYLON.MeshBuilder.CreateBox("box", { width: 0.2, height: 1.0, depth: 0.2 }, scene);

  // Set the box's initial position
  root.position = new BABYLON.Vector3(0, 0.5, 0); // The bottom of the box rests on y=0

  // Set the box's initial rotation
  root.rotationQuaternion = new BABYLON.Quaternion(0, 0, 0, 1);

  // Create left and right "hand" spheres
  var leftHand = BABYLON.MeshBuilder.CreateSphere("leftHand", { diameter: 0.15 }, scene);
  var rightHand = BABYLON.MeshBuilder.CreateSphere("rightHand", { diameter: 0.15 }, scene);

  var defaultMaterial = new BABYLON.StandardMaterial("defaultMat", scene);
  var graspMaterial = new BABYLON.StandardMaterial("graspMat", scene);

  // 2. Assign a color to the material's diffuseColor property
  graspMaterial.diffuseColor = new BABYLON.Color3(0, 1, 0); // RGB for red

  // Position the hands in front of the box (avatar)
  var handOffset = 0.2;
  var handHeight = 0.7;
  var handOffsetY = handHeight - root.scaling.y / 2; // Offset to center the spheres

  leftHand.position = new BABYLON.Vector3(-handOffset, handOffsetY, handOffset);
  rightHand.position = new BABYLON.Vector3(handOffset, handOffsetY, handOffset);

  // Parent the hands to the box (avatar)
  leftHand.parent = root;
  rightHand.parent = root;

  // Create the action manager for the scene
  scene.actionManager = new BABYLON.ActionManager(scene);

  const walkSpeed = 2;
  const rotSpeed = Math.PI * 1.0;
  const handSpeed = 1.0;
  // relative to origin of avatar, which is currently at pelvis height
  const minHandHeight = -1.0
  const maxHandHeight = 1.0

  isInXR = function() {
    // xrHelper is a global defined elsewhere
    return xrHelper && xrHelper.baseExperience.state === BABYLON.WebXRState.IN_XR && xrHelper.input && xrHelper.input.controllers;
  }

  avatarUpdate = function (deltaTime) {

    // update parenting based on whether we're in VR
    if (!isInXR()) {
      // Ensure hands are parented to root when not in WebXR mode
      leftHand.parent = root;
      rightHand.parent = root;
      root.isVisible = true;

      // Position the hands in front of the box (avatar)
      var handOffset = 0.2;
      var handHeight = 0.7;
      var handOffsetY = handHeight - root.scaling.y / 2; // Offset to center the spheres

      leftHand.position = new BABYLON.Vector3(-handOffset, handOffsetY, handOffset);
      rightHand.position = new BABYLON.Vector3(handOffset, handOffsetY, handOffset);

    } else {
      // If in XR mode, hands should not be parented to the root
      leftHand.parent = null;
      rightHand.parent = null;
      root.isVisible = false;
      // hide root
    }
        
    if (!isInXR()) {
        // Don't use inputMgr.getKeyDown or getKeyUp here! you will get duplicate events.
      // Use them in avatarHandleInputEvents.

      // Move the box based on key states
      if (inputMgr.getKey("w")) {
        root.translate(BABYLON.Axis.Z, walkSpeed * deltaTime, BABYLON.Space.LOCAL);
      } else if (inputMgr.getKey("s")) {
        root.translate(BABYLON.Axis.Z, -walkSpeed * deltaTime, BABYLON.Space.LOCAL);
      }

      // Rotate the box based on key states
      if (inputMgr.getKey("a")) {
        root.rotate(BABYLON.Axis.Y, rotSpeed * deltaTime, BABYLON.Space.LOCAL);
      } else if (inputMgr.getKey("d")) {
        root.rotate(BABYLON.Axis.Y, -rotSpeed * deltaTime, BABYLON.Space.LOCAL);
      }

      // Move hands up and down based on key states
      if (inputMgr.getKey("e")) {
        leftHand.position.y += handSpeed * deltaTime;
        rightHand.position.y += handSpeed * deltaTime;
        
        // Cap hand height to max
        leftHand.position.y = Math.min(leftHand.position.y, maxHandHeight);
        rightHand.position.y = Math.min(rightHand.position.y, maxHandHeight);
      } else if (inputMgr.getKey("q")) {
        leftHand.position.y -= handSpeed * deltaTime;
        rightHand.position.y -= handSpeed * deltaTime;

        // Cap hand height to min
        leftHand.position.y = Math.max(leftHand.position.y, minHandHeight);
        rightHand.position.y = Math.max(rightHand.position.y, minHandHeight);
      }    
    }
  }

  avatarHandleInputEvents = function () {
    // update hand color; perf todo: is this slow?
    if (inputMgr.getKeyDown("f") || inputMgr.getButtonDown(0)) {
      leftHand.material = graspMaterial
      rightHand.material = graspMaterial;
    } else if (inputMgr.getKeyUp("f") || inputMgr.getButtonUp(0)) {
      leftHand.material = defaultMaterial
      rightHand.material = defaultMaterial;
    }
  }

  // XR Controller Setup
  scene.onAfterRenderObservable.add(() => {
    // Use the WebXR experience helper to get controllers
    if (isInXR()) {
      const xr = xrHelper;
      handIndex = 0;
      xr.input.controllers.forEach(controller => {
        if (handIndex <= 1) {
          // todo: figure out how to tell which controller is left/right
          const handObj = (handIndex == 0) ? leftHand : rightHand;
          handObj.position = controller.pointer.position.clone();
          handObj.rotationQuaternion = controller.pointer.rotationQuaternion.clone();
          handIndex++;
        }
      });
    }
  });

  // Headset Pose Update: Simply update the avatar's position based on the camera's position.
  scene.onBeforeRenderObservable.add(() => {
    if (scene.activeCamera.isVR) {
      root.position = scene.activeCamera.position.clone();
    }
  });  

  const avatarObj = {
    root: root,
    hands: [leftHand, rightHand],
    update: avatarUpdate,
    handleInputEvents: avatarHandleInputEvents
  }
  return avatarObj;
}
