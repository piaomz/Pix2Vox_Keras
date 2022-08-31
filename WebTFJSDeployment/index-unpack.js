
var camera, scene, renderer,
  model, loader, stats, controls;
//browserify index-unpack.js -o index.js
array=false;

var ratioof3dview=0.8
function init(array) {
  
  // Stats
  stats = new Stats();
  document.body.appendChild(stats.dom);

  // Setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xFFFFFF);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth*ratioof3dview, window.innerHeight, false);

  let mainNode = document.getElementById("viewport");
  mainNode.appendChild(renderer.domElement);

  camera = new THREE.PerspectiveCamera(40, window.innerWidth*ratioof3dview / window.innerHeight, 1, 1000);
  //camera.position.set(20, 4, 10);
  camera.position.set(60, 30, 30);

  // Camera Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.update();

  controls.addEventListener('start', function () {
    controls.autoRotate = false;
  });

  // Lights
  var ambientLight = new THREE.AmbientLight(0x666666);
  scene.add(ambientLight);

  var light = new THREE.DirectionalLight(0xdfebff, 1);
  light.position.set(300, 500, 100);
  light.position.multiplyScalar(1);
  scene.add(light);



  models = {
    'mymodel': './models/modelt.binvox',
    'mymodel1': './models/predict01.binvox',
    'mymodel2': './models/predict03.binvox'
  };
  
  
  // GUI
  params = {
    model: models["mymodel"],
    size: 0.93,
    material: {
      color: 0xbdbdbd,
    },
    LOD: {
      maxPoints: 1,
      maxDepth: 10
    },
    renderer: {
      triangles: 0
    }
  };

  var gui = new dat.GUI();

  modelselector=gui.add(params, 'model', models)
  modelselector.onChange(m => {
    _toggleLoading(true);
    loader.loadFile(params.model, function (voxels) {
      resetScene();
      model = voxels;
      renderModel();
    });
  });

  gui.add(params, 'size').min(0.01).max(1).step(0.01).onFinishChange(d => {
    params.size = d;
    loader.setVoxelSize(params.size);
    resetScene();
    model = loader.generateMesh(loader.octree);
    renderModel();
  });

  var lod = gui.addFolder('Level Of Detail');
  lod.add(params.LOD, 'maxPoints').min(1).max(30).step(1).onFinishChange(d => {
    params.LOD.maxPoints = d;
    resetScene();
    loader.setLOD(d);
    _toggleLoading(true);
    loader.update().then((octree) => {
      model = loader.generateMesh(octree);
      renderModel();
    });
  });
  lod.add(params.LOD, 'maxDepth').min(1).max(10).step(1).onFinishChange(d => {
    params.LOD.maxDepth = d;
    resetScene();
    loader.setLOD(undefined, d);
    _toggleLoading(true);
    loader.update().then((octree) => {
      model = loader.generateMesh(octree);
      renderModel();
    });
  });
  lod.open();

  var mat = gui.addFolder('Material');
  mat.addColor(params.material, 'color').onChange(color => {
    params.material.color = color;
    model.material.color.set(color)
    requestRenderIfNotRequested();
  });
  var info = gui.addFolder('Render Info');
  info.add(renderer.info.render, 'triangles').listen();


  // Voxel Loader
  loader = new VoxelLoader();
  loader.setVoxelSize(params.size);
  loader.setLOD(params.LOD.maxPoints, params.LOD.maxDepth);

  loader.loadFile(params.model, function (voxels) {
    model = voxels;
    renderModel()
  });

  function _toggleLoading(bool) {
    let loaderNode = document.getElementById("loader");
    if (bool) {
      loaderNode.style.display = "block";
    } else {
      loaderNode.style.display = "none";
    }
    function wait() {
      if (loaderNode.style.display != "block") {
        requestAnimationFrame(wait);
      }
    }
    wait();
  }

  function renderModel() {
    _toggleLoading(false);
    model.position.x = 0.5
    model.position.z = 0.5
    scene.add(model)
    requestRenderIfNotRequested()
  }

  function resetScene() {
    scene.remove(model);
    model.geometry.dispose()
    scene.dispose();
    requestRenderIfNotRequested();
  }

    
  predict = async function () {
    await _toggleLoading(true);
    imgdataraw=await cv.imread('imgview')
    //console.log(imgdata.size()['width'],imgdata.size()['height'])
    var imgdata1 = new cv.Mat();
    var dsize = new cv.Size(224, 224);
    cv.resize(imgdataraw, imgdata1, dsize, 0, 0, cv.INTER_AREA);
    var rect= new cv.Rect(0,0,223,223);
    imgdata1 = imgdata1.roi(rect);
    var imgdata = new cv.Mat();
    cv.resize(imgdata1, imgdata, dsize, 0, 0, cv.INTER_AREA);
    //imgdata=imgdata1
    /*

    for(col=0;col<imgdata.size()['width'];col++){
      var rows=[];
      for(row=0;row<imgdata.size()['height'];row++){
          pixel=imgdata.ucharPtr(col,row);
          var rgb=[];
          //var alpha=pixel[3]/255
          rgb.push(pixel[0])
          rgb.push(pixel[1])
          rgb.push(pixel[2])
          rgb.push(pixel[3])
          rows.push(rgb)
      }
      imgarraytest.push(rows)
    }
*/
    imgarray=[]
    for(col=0;col<imgdata.size()['width'];col++){
      var rows=[];
      for(row=0;row<imgdata.size()['height'];row++){
          pixel=imgdata.ucharPtr(col,row);
          var rgb=[];
          var alpha=1;
          if(pixel[3]==0){
            alpha=0;
          }else{
            alpha=pixel[3]/255;
          }
          rgb.push(((pixel[0]*alpha+(1-alpha)*225)/255-0.5)/0.5)
          rgb.push(((pixel[1]*alpha+(1-alpha)*225)/255-0.5)/0.5)
          rgb.push(((pixel[2]*alpha+(1-alpha)*225)/255-0.5)/0.5)
          rows.push(rgb)
      }
      imgarray.push(rows)
    }
    console.log(imgarray)
    imgtensor = tf.tensor([imgarray]);
    //imgtensor = tf.tensor(atest);
    var dateBegin = new Date();
    prediction = await kerasmodel.predict(imgtensor);
    var dateEnd = new Date();
    var dateDiff = dateEnd.getTime() - dateBegin.getTime();
    document.getElementById("inferencetime").innerHTML = String(dateDiff)+'ms';
    predvoxels=[];
    await prediction.array().then(
        function(predarray){
          console.log(predarray);
          for(col1=0;col1<32;col1++){
            for(col2=0;col2<32;col2++){
              for(col3=0;col3<32;col3++){
                if(predarray[0][col3][col2][col1][0]>0.35){
                  predvoxels.push({ x: col1, y: col3, z: col2 });
                }
              }
            }
          }
          //console.log(predvoxels)
        }
    )   
    builder = new BINVOX.Builder()
    buildprem={ 
      dimension: { depth: 32, width: 32, height: 32 },
      translate: { depth: -0.354112, width: -0.113191, height: -0.334354 },
      scale: 0.708224,
      voxels: predvoxels
      }
    console.log(buildprem);
    voxfile = builder.build(buildprem)
    data = new Blob([voxfile],{ type: 'application/octet-stream'});
    downloadUrl = window.URL.createObjectURL(data);
    models = {
      'mymodel': downloadUrl,
    };
    params = {
      model: models["mymodel"],
      size: 0.93,
      material: {
        color: 0xbebebe,
      },
      LOD: {
        maxPoints: 1,
        maxDepth: 10
      },
      renderer: {
        triangles: 0
      }
    }
    modelselector.updateDisplay();
    //console.log(params.model);
    loader.loadFile(params.model, function (voxels) {
      resetScene();
      model = voxels;
      renderModel();
    }); 
  }
}

function resizeRendererToDisplaySize() {
  const canvas = renderer.domElement;
  const height = window.innerHeight;
  const width = window.innerWidth*ratioof3dview;
  const needResize = canvas.width != width || canvas.height != height;
  if (needResize) {
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    requestRenderIfNotRequested()
  }
}

let renderRequested = false;
function render() {
  renderRequested = false;
  controls.update();

  stats.begin();
  renderer.render(scene, camera);
  stats.end();
}

function requestRenderIfNotRequested() {
  if (!renderRequested) {
    renderRequested = true;
    requestAnimationFrame(render);
  }
}



async function loadmodel(){
  tf = require("@tensorflow/tfjs")
  kerasmodel = await tf.loadLayersModel('jsmodel_small/model.json');
  console.log("success load model");
  document.getElementById("loadmodelview").innerHTML = "模型加载成功!";
  testarray=[]
  for(testf=0;testf<224;testf++){

  }
  testarray.push()
  //img = document.getElementById('img')
  //image = tf.fromPixels(img);  // for example
  //prediction = model.predict(image);
}


window.addEventListener('load', function () {
  loadmodel();
  init(false);
  render();

  controls.addEventListener('change', requestRenderIfNotRequested);
  window.addEventListener('resize', resizeRendererToDisplaySize);
})

