class InputManager {
  constructor() {  // Note: We now require the Babylon.js scene
    this.keyHeld = {};
    this.keyDown = {};
    this.keyUp = {};
    this.buttonHeld = {};  // VR controller button states
    this.buttonDown = {};
    this.buttonUp = {};

    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleKeyUp = this.handleKeyUp.bind(this);

    window.addEventListener("keydown", this.handleKeyDown);
    window.addEventListener("keyup", this.handleKeyUp);
  }

  handleKeyDown(event) {
    if (!this.keyHeld[event.key]) { 
      this.keyHeld[event.key] = true;
      this.keyDown[event.key] = true;
    }
  }

  handleKeyUp(event) {
    if (this.keyHeld[event.key]) { 
      this.keyHeld[event.key] = false;
      this.keyUp[event.key] = true;
    }
  }

  handleVRButtonDown(buttonId) {
    if (!this.buttonHeld[buttonId]) { 
      this.buttonHeld[buttonId] = true;
      this.buttonDown[buttonId] = true;
    }
  }

  handleVRButtonUp(buttonId) {
    if (this.buttonHeld[buttonId]) { 
      this.buttonHeld[buttonId] = false;
      this.buttonUp[buttonId] = true;
    }
  }

  addVRListeners(xrHelper) {
    xrHelper.input.onControllerAddedObservable.add((controller) => {
      controller.onMotionControllerInitObservable.add((motionController) => {
        if (motionController.handness === 'left') {
          const xr_ids = motionController.getComponentIds();
          let triggerComponent = motionController.getComponent(xr_ids[0]);//xr-standard-trigger
          triggerComponent.onButtonStateChangedObservable.add(() => {
            if (triggerComponent.changes.pressed) {
              // is it pressed?
              const buttonId = 0; // todo
              if (triggerComponent.pressed) {
                this.handleVRButtonDown(buttonId);
              } else {
                this.handleVRButtonUp(buttonId);
              }
            }
          });
        }
      })
    });
  }

  getSerializeState() {
    const obj = {
      keyHeld: this.keyHeld,
      keyDown: this.keyDown,
      keyUp: this.keyUp,
      buttonHeld: this.buttonHeld,
      buttonDown: this.buttonDown,
      buttonUp: this.buttonUp
    };
    return obj;
  }

  // returns true is the key is currently down
  getKey(key) {
    return !!this.keyHeld[key];  // returns true if key is pressed, otherwise false
  }

  // returns true if key was pressed on this frame
  getKeyDown(key) {
    return !!this.keyDown[key];
  }

  // returns true if key was released on this frame
  getKeyUp(key) {
    return !!this.keyUp[key];
  }

  getButton(buttonId) {
    return !!this.buttonHeld[buttonId];  // returns true if key is pressed, otherwise false
  }

  // returns true if key was pressed on this frame
  getButtonDown(buttonId) {
    return !!this.buttonDown[buttonId];
  }

  // returns true if key was released on this frame
  getButtonUp(buttonId) {
    return !!this.buttonUp[buttonId];
  }
  
  onEndFrame() {
    this.keyUp = {};
    this.keyDown = {};
    this.buttonUp = {};
    this.buttonDown = {};
  }

  dispose() {
    window.removeEventListener("keydown", this.handleKeyDown);
    window.removeEventListener("keyup", this.handleKeyUp);
    // Note: Since the VR controller events are attached directly to the scene, 
    // you'd need to ensure the scene is disposed of properly to clean up those event listeners.
  }
}



