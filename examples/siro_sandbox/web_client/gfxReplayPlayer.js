var loadedMeshes = {};
var loads = {}
var keyframeQueue = [];
var isProcessing = false;
var recentNumLoads = -1;


function remapFilepath(filepath) {

    return filepath;

    // if (filepath.includes("/objects/")) {
    //     result = "data/hitl/simplified" + filepath.slice(4)
    //     return result
    // } else {
    //     return filepath
    // }
}

function areArraysEqual(arr1, arr2) {
    return arr1.length === arr2.length && arr1.every((value, index) => value === arr2[index]);
}
  
function handleFrame(node, frame) {

    if (areArraysEqual(frame.up, [0, 1, 0])) {
        return node;
    } else if (areArraysEqual(frame.up, [0, 0, 1])) {
        // rotate 90 about x axis
        node.rotate(BABYLON.Axis.X, -Math.PI / 2, BABYLON.Space.LOCAL);
        let newRootNode = new BABYLON.TransformNode("root2");
        node.parent = newRootNode;
        return newRootNode;
    } else {
        assert(false, `unexpected value for frame.up: ${frame.up}`);
        return null;
    }
}


function loadMeshesFromCreations(keyframe) {

    if (keyframe.loads) {
        for (const load of keyframe.loads) {
            if (load.filepath in loads) {
                // skip
            } else {
                loads[load.filepath] = load;
            }
        }
    }

    if (!("creations" in keyframe)) {
        // do nothing
    } else {
        promises = keyframe.creations.map(creationWrapper => {
            let creation = creationWrapper.creation;
            if (!(creation.filepath in loads)) {
                console.warn(`Couldn't create ${creation.filepath} because there was no corresponding load`);
                loadedMeshes[load.filepath] = null; // indicate failed load
                return Promise.resolve();
            }
            assert(creation.filepath in loads, `didn't find ${creation.filepath} in loads`);
            load = loads[creation.filepath];
            // console.log(`ImportMeshAsync ${load.filepath}`);
            if (load.filepath in loadedMeshes) {
                return Promise.resolve(); // already loaded; do nothing
            } else {
                recentNumLoads++;
                remappedFilepath = remapFilepath(load.filepath);

                // We declare a callback to process the imported mesh. We need to
                // "capture by value" the load variable into this callback. In
                // javascript, this trick is called IIFE.
                var importFinishedCallback = ((loadCapture) => {
                    return (result) => {
                        // console.log(loadCapture.filepath);

                        if (result.meshes.length == 0) {
                            console.log(`failed to load any nodes for ${loadCapture.filepath}`);
                            loadedMeshes[loadCapture.filepath] = null; // indicate failed load
                        } else {
                            let rootNode = new BABYLON.TransformNode("root");
                            result.meshes.forEach(mesh => {
                                mesh.parent = rootNode;
                                mesh.setEnabled(false);  // Hide the original mesh
                            });

                            rootNode = handleFrame(rootNode, loadCapture.frame);

                            loadedMeshes[loadCapture.filepath] = rootNode;
                        }
                    };
                })(load);

                var errorCallback = ((loadCapture) => {
                    return (error) => {
                        console.error(`Error loading mesh:`, error.message);
                        console.error(`  loadCapture.filepath: ${loadCapture.filepath}`);
                        loadedMeshes[loadCapture.filepath] = null;  // indicate failed load
                    };
                })(load);

                return BABYLON.SceneLoader.ImportMeshAsync("", "", remappedFilepath, scene).then(
                    importFinishedCallback).catch(errorCallback);;
            }
        });
    }

    if (recentNumLoads > 0) {
        console.log(`loading ${recentNumLoads} assets...`);
    }

    return Promise.all(promises);

}

/*
function loadMeshes(keyframe) {
    let promises = []
    recentNumLoads = 0
    if (!("loads" in keyframe)) {
        // do nothing
    } else {
        promises = keyframe.loads.map(load => {
            // console.log(`ImportMeshAsync ${load.filepath}`);
            if (load.filepath in loadedMeshes) {
                return Promise.resolve(); // already loaded; do nothing
            } else {
                recentNumLoads++;
                remappedFilepath = remapFilepath(load.filepath);
                return BABYLON.SceneLoader.ImportMeshAsync("", "", remappedFilepath, scene).then(result => {

                    // IIFE weirdness to capture load from outside this function
                    localLoad = ((x) => { return x; })(load);
                    console.log(localLoad.filepath);

                    if (result.meshes.length == 0) {
                        console.log(`failed to load any nodes for ${localLoad.filepath}`);
                        loadedMeshes[localLoad.filepath] = null; // indicate failed load
                    } else {
                        let rootNode = new BABYLON.TransformNode("root");
                        result.meshes.forEach(mesh => {
                            mesh.parent = rootNode;
                            mesh.setEnabled(false);  // Hide the original mesh
                        });
                        // rootNode = handleFrame(load.frame, rootNode); // doesn't work because load isn't available here

                        // Use IIFE to capture the 'load' object in the closure
                        rootNode = handleFrame(rootNode, localLoad.frame);

                        loadedMeshes[localLoad.filepath] = rootNode;
                    }
                }).catch(error => {
                    console.error(`Error loading mesh:`, error.message);
                    // You can add additional handling here, e.g., displaying a user-friendly message.
                    loadedMeshes[load.filepath] = null;  // indicate failed load
                });;
            }
        });
    }

    if (recentNumLoads > 0) {
        console.log(`loading ${recentNumLoads} assets...`);
    }

    return Promise.all(promises);
}
*/

let instanceCounter = 0;

var createdInstances = [];
var recentNumMeshInstances = 0;
var recentNumClonedNodes = 0;
var recentNumVerts = 0;

let instancesByKey = {};  // Map from instanceKey to the specific instance for easy lookup during stateUpdates

function instanceNodeHierarchy(node, parentNode = null) {
    let newInstance;

    // todo: avoid naming conflict (but not required for Babylon)
    // Check if the node is a mesh and has geometry
    if (node instanceof BABYLON.Mesh && node.geometry) {
        newInstance = node.createInstance(node.name + "_instance_" + instanceCounter++);
        recentNumMeshInstances++;
        recentNumVerts += newInstance.getTotalVertices();
    } else {
        // Clone for non-mesh nodes or meshes without geometry
        newInstance = node.clone(node.name + "_instance_" + instanceCounter++);
        recentNumClonedNodes++;
    }

    createdInstances.push(newInstance);

    // Copy transformations
    newInstance.position = node.position.clone();
    if (node.rotationQuaternion) {
        newInstance.rotationQuaternion = node.rotationQuaternion.clone();
    } else {
        newInstance.rotation = node.rotation.clone();
    }
    newInstance.scaling = node.scaling.clone();

    // If there's a specified parent, set it
    if (parentNode) {
        newInstance.parent = parentNode;
    }

    // Recursively instance children
    node.getChildren().forEach(child => {
        instanceNodeHierarchy(child, newInstance);
    });

    return newInstance;
}

function poseMeshes(keyframe) {

    if ("creations" in keyframe) {
        recentNumMeshInstances = 0;
        recentNumClonedNodes = 0;
        recentNumVerts = 0;
        numRenderInstances = 0;
        keyframe.creations.forEach(creationWrapper => {
            if (loadedMeshes[creationWrapper.creation.filepath]) {
                // Instance the rootNode and store it using instanceKey
                // let instance = loadedMeshes[creationWrapper.creation.filepath].createInstance("instance" + creation.instanceKey);
                let tmpTotalVerts = recentNumVerts;
                let instance = instanceNodeHierarchy(loadedMeshes[creationWrapper.creation.filepath])
                console.log(`instance ${creationWrapper.creation.filepath} with ${recentNumVerts - tmpTotalVerts} verts`);
                
                if (creationWrapper.creation.scale) {
                    var scale = BABYLON.Vector3.FromArray(creationWrapper.creation.scale);
                    instance.scaling = scale;
                }

                instancesByKey[creationWrapper.instanceKey] = instance;
                numRenderInstances++;
            }
        });
        console.log(`Created ${numRenderInstances} instances`);
        console.log(`${recentNumClonedNodes} empty nodes, ${recentNumMeshInstances} mesh instances, ${recentNumVerts} vertices`);
    }

    if ("stateUpdates" in keyframe) {
        keyframe.stateUpdates.forEach(update => {
            let specificInstance = instancesByKey[update.instanceKey];

            if (specificInstance && update.state.absTransform) {
                if (update.state.absTransform.translation) {
                    specificInstance.position = BABYLON.Vector3.FromArray(update.state.absTransform.translation);
                }
                if (update.state.absTransform.rotation) {
                    let quat_wxyz = update.state.absTransform.rotation;
                    let quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]];
                    specificInstance.rotationQuaternion = BABYLON.Quaternion.FromArray(quat_xyzw);
                }
            }
        });
    }
}

function processNextKeyframe() {
    if (keyframeQueue.length === 0) {
        isProcessing = false;
        return;
    }
    
    isProcessing = true;
    var currentKeyframe = keyframeQueue.shift();
    
    loadMeshesFromCreations(currentKeyframe).then(() => {
        if (recentNumLoads > 0) {
            console.log(`Done loading ${recentNumLoads} assets`);
            recentNumLoads = -1;
        }
        poseMeshes(currentKeyframe);
        processNextKeyframe();
    });
}

function addKeyframeToQueue(keyframe) {
    keyframeQueue.push(keyframe);
    if (!isProcessing) {
        processNextKeyframe();
    }
}

function deleteAllInstancesFromKeyframes() {
    createdInstances.forEach(instance => {
        instance.dispose();
    });

    // Clear the instances array
    createdInstances = [];
    instancesByKey = {};
}





// ... (initial part of the code remains unchanged)


// ... (the rest of the code remains unchanged)
