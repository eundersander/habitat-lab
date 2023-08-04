
//var button;
var text1;
var panel;
var sceneRef;

var disable = false;

function createGuiText(scene) {

    if (disable) {
        return;
    }

    sceneRef = scene;

    var manager = new BABYLON.GUI.GUI3DManager(scene);

    panel = new BABYLON.GUI.PlanePanel();
    panel.margin = 0.2;

    manager.addControl(panel);

    var anchor = new BABYLON.TransformNode("");
    panel.linkToTransformNode(anchor);
    panel.position.x = 0;
    panel.position.y = 1;
    panel.position.z = 0;
    // just big enough for one unit-size button
    //panel.size = new BABYLON.Vector3(1, 1, 1);   
    panel.rows = 1;
    panel.columns = 1;
    panel.scaling = new BABYLON.Vector3(0.4, 0.4, 0.4);

    // note I can't find a way to set the 3D size of the button!
    var button = new BABYLON.GUI.Button3D("gui text");
    panel.addControl(button);

    text1 = new BABYLON.GUI.TextBlock();
    text1.text = "[gui text]";
    text1.color = "white";
    text1.fontSize = 77;
    text1.fontFamily = "Arial";
    button.content = text1;

    // button = new BABYLON.GUI.HolographicButton("orientation");
    // button.text = "[gui text]";  // You can set text directly on the holographic button
    // button.color = "white";
    // panel.addControl(button);
    

}

isInXR = function() {
    // xrHelper is a global defined elsewhere
    return xrHelper && xrHelper.baseExperience.state === BABYLON.WebXRState.IN_XR && xrHelper.input && xrHelper.input.controllers;
}

function updateGuiText(str) {

    if (disable) {
        return;
    }
    
    assert(text1);
    text1.text = str;

    // Calculate the forward direction from the camera
    var forward = sceneRef.activeCamera.getForwardRay().direction;
    forward.normalize();
    let scale = 1;
    if (isInXR()) {
        scale = -1;
    }
    var up = scene.activeCamera.upVector;
    //var right = scene.activeCamera.leftVector;
    
    // Position the panel 2 units in front of the camera
    var dist = 4;
    var desiredPosition = sceneRef.activeCamera.position.add(forward.scale(dist * scale));

    // Offset the panel to the top-left corner of the screen.
    // Adjust the scale values to control the offset.
    //desiredPosition = desiredPosition.subtract(right.scale(dist * 0.8)); // move to the left
    desiredPosition = desiredPosition.add(up.scale(dist * 0.3));         // move upwards

    panel.position = desiredPosition;

    // Make the panel face the camera
    //panel.lookAt(sceneRef.activeCamera.position);
    panel.node.lookAt(sceneRef.activeCamera.position);
    // panel.orientation = BABYLON.GUI.PlanePanel.FACEORIGIN_ORIENTATION;

}