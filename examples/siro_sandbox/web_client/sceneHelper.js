// sceneHelper.js


function createGridLines(scene) {

  // Grid lines
  const gridSize = 1;
  const cellSize = 0.1;
  const halfSize = gridSize / 2;

  const colorX0 = new BABYLON.Color3(0.5, 0.0, 0.0);
  const colorX1 = new BABYLON.Color3(1.0, 0.0, 0.0);
  const colorZ0 = new BABYLON.Color3(0.0, 0.0, 0.5);
  const colorZ1 = new BABYLON.Color3(0.0, 0.0, 1.0);
  const xStart = -halfSize;
  const xEnd = halfSize;
  const zStart = -halfSize;
  const zEnd = halfSize;

  for (let x = xStart; x <= xEnd; x += cellSize) {
      const points = [
          new BABYLON.Vector3(x, 0, zStart),
          new BABYLON.Vector3(x, 0, zEnd)
      ];

      const line = BABYLON.MeshBuilder.CreateLines("xLine" + x, { points: points }, scene);
      line.color = x == xEnd ? colorX1 : colorX0;
  }

  for (let z = zStart; z <= zEnd; z += cellSize) {
      const points = [
          new BABYLON.Vector3(xStart, 0, z),
          new BABYLON.Vector3(xEnd, 0, z)
      ];

      const line = BABYLON.MeshBuilder.CreateLines("zLine" + z, { points: points }, scene);
      line.color = z == zEnd ? colorZ1 : colorZ0;
  }
}

function loadTestModel(scene) {

    // URL to your .glb file
  // const glbURL = "105515211_173104173.glb";
  //const glbURL = "data/fpss/fphab/stages/103997919_171031233.glb"
  const glbURL = "data/objects/ycb/meshes/003_cracker_box/textured.glb"

  // Load the .glb file and log progress
  BABYLON.SceneLoader.ImportMesh("", glbURL, "", scene, 
    (meshes) => {
        // Called when the file has finished loading and meshes are available
        console.log(`loaded ${meshes.length} meshes!`);

        //meshes[0].position = new BABYLON.Vector3(15, 0, 8);
        meshes[0].position = new BABYLON.Vector3(0, 0.1, 0);

        if (false) {
          // // Define the simplification setting for 10% quality
          // var simplificationSetting = new BABYLON.SimplificationSettings(0.1, 0);

          vertSimplifyThreshold = 10000;
          vertCullThreshold = 33;
          // var meshesToSimplify = [];
          // for (var mesh of meshes) {
          //   if (mesh.getTotalVertices() > vertSimplifyThreshold) {
          //     meshesToSimplify.push(mesh);
          //   }
          // }

          // // Counter for the number of meshes that have completed simplification
          // var completedMeshes = 0;

          // // Function to handle the callback after each mesh is simplified
          // function onSimplified() {
          //   completedMeshes++;
          //   console.log(`Finished simplifying ${completedMeshes} of ${meshesToSimplify.length}`);
            
          //   if (completedMeshes === meshesToSimplify.length) {
          //     // All meshes are simplified, calculate total vertex count
          //     var totalVertexCount = 0;
          //     for (var mesh of meshes) {
          //       if (mesh) {
          //         totalVertexCount += mesh.getTotalVertices();
          //       }
          //     }

          //     console.log("Total vertex count after simplification:", totalVertexCount);
          //   }
          // }

          // for (var mesh of meshesToSimplify) {
          //   mesh.simplify([simplificationSetting], true, BABYLON.SimplificationType.QUADRATIC, onSimplified);
          // }

          let vertCount = 0;
          let meshCount = 0;
          for (let i = 0; i < meshes.length; i++) {
            mesh = meshes[i];
            mesh.isPickable = false;

            if (mesh.getTotalVertices() < vertCullThreshold ||
              mesh.getTotalVertices() > vertSimplifyThreshold) {
              mesh.isVisible = false;
            } else {
              meshCount++;
              vertCount += mesh.getTotalVertices();
            }

            // if (mesh.material) {
            //   var pbrMaterial = mesh.material;

            //   // Disabling PBR specific textures
            //   pbrMaterial.albedoTexture = null;
            //   pbrMaterial.reflectivityTexture = null;
            //   pbrMaterial.microSurfaceTexture = null;
            //   pbrMaterial.bumpTexture = null;
            //   pbrMaterial.metallicTexture = null; // if you're using metallic/roughness workflow
            //   pbrMaterial.roughnessTexture = null; // if you're using metallic/roughness workflow
            //   // ... and so on for other textures specific to PBR
                          
            // }          
          }

          console.log(`visible: ${meshCount} meshes, ${vertCount} vertices`);
        }
      
    }, 
    (event) => {
        // Track the download progress
        if (event.lengthComputable) {
            const progress = (event.loaded * 100 / event.total).toFixed(2);
            console.log(`Downloading: ${progress}%`);
        }
    }, 
    (scene, message, exception) => {
        // Handle any errors
        console.error("Error loading the .glb file:", message, exception);
    }
  );

}

function createScene(engine, canvas, numCubes) {
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color3.Black;

  scene.useRightHandedSystem = true;

  const alpha = Math.PI / 4;
  const beta = Math.PI / 3; // Math.PI / 2;
  const radius = 20;
  const target = new BABYLON.Vector3(0, 1, 0);

  const camera = new BABYLON.ArcRotateCamera("Camera", alpha, beta, radius, target, scene);
  camera.attachControl(canvas, true);
  camera.wheelPrecision=50

  const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(1, 1, 0));

  const cubes = [];

  const boxSize = 0.1; // Adjust the size of the boxes as desired

  if (false) {
    for (let i = 0; i < numCubes; i++) {

        // Define the extents to match cheezit box
        const xMin = -0.04;
        const xMax = 0.01;
        const yMin = 0.0;
        const yMax = 0.2;
        const zMin = -0.09;
        const zMax = 0.06;

        // Calculate the dimensions of the box
        const width = xMax - xMin;
        const height = yMax - yMin;
        const depth = zMax - zMin;

        // Create the box with its dimensions centered at the origin
        const childBox = BABYLON.MeshBuilder.CreateBox("box", { width, height, depth }, scene);
        const boxMaterial = new BABYLON.StandardMaterial("material" + i, scene);
        boxMaterial.diffuseColor = BABYLON.Color3.Random();
        childBox.material = boxMaterial;

        // Calculate the center offset
        const centerX = (xMin + xMax) / 2;
        const centerY = (yMin + yMax) / 2;
        const centerZ = (zMin + zMax) / 2;

        // Apply the center offset to the box
        childBox.position = new BABYLON.Vector3(centerX, centerY, centerZ);

        // oops, we don't need this transform, because Habitat recenters the object origin to the extent center
        // const transformNode = new BABYLON.TransformNode("transformNode", scene);
        // childBox.setParent(transformNode);
        transformNode = childBox;

        transformNode.position.x = 0.5 + i * 0.1; // Adjust the initial position for each cube
        transformNode.position.y = 1 + i * 0.1;

        cubes.push(transformNode);
    }
  }

  const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 25, height: 25 });
  ground.visibility = false;
  // createGridLines(scene);
  // const ground = null;

  // loadTestModel(scene)

  // note that ground mesh is needed for VR
  return [scene, ground, cubes];
}

