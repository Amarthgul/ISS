import * as THREE from 'three';
import { TrackballControls } from "https://cdn.skypack.dev/three-trackballcontrols-ts@0.2.3";


// ---------------------------------------------------------------------
// ------------------------------- Settings ----------------------------
// ---------------------------------------------------------------------

// When set to true, use fbx loader, otherwise obj loader 
const useFBX = true;

// When set to false, texture maps will not be loaded 
const useTextures = false; 

// The main model to be loaded 
const modelToUse = null;

// The texture of the main model 
const textureToUse = null;

// ---------------------------------------------------------------------
// ------------------------------ Functions ----------------------------
// ---------------------------------------------------------------------

export class RenderCanvas {

    #canvas;             // The primary element of the rendering target 
    #renderer;           // WebgL renderer

    #backgroundColor = 0x000000;
    #backgroundOpacity = 0; 

    #scene; 

    #camera;             // Main camera 
    #camera135FocalLength = 50; // 50mm lens 
    #cameraAoV;          // Field of view in degree 
    #cameraAspect = 2;   // Width / height ratio
    #cameraNear = 0.1;   // Near clipping plane 
    #cameraFar = 2000;   // Far clipping plane

    #controls;           // Trackball interactive control 

    #manualMovement = true; 

    // ---------------------------------------------------------------------
    // Class constructor 
    constructor(canvasID, parentID) {
        this.#canvas = document.querySelector(canvasID);
        this.#renderer = new THREE.WebGLRenderer({
            antialias: true,
            canvas: this.#canvas
        });

        this.#renderer.setClearColor(this.#backgroundColor, this.#backgroundOpacity);
        document.getElementById(parentID).appendChild(this.#renderer.domElement); 
    }

    // ---------------------------------------------------------------------
    // Create the scene, set up initial camera 
    startup() {

        // Set up the camera 
        this.#cameraAoV = std135Aov(this.#camera135FocalLength);
        this.#camera = new THREE.PerspectiveCamera(
            this.#cameraAoV,
            this.#cameraAspect,
            this.#cameraNear,
            this.#cameraFar);
        this.#camera.position.z = 100;

        this.#scene = new THREE.Scene();

        // Create trackball which allows the user to rotate, pan, zoom 
        this.#controls = new TrackballControls(this.#camera, this.#renderer.domElement);
        this.#controls.rotateSpeed = 4;
        this.#controls.dynamicDampingFactor = 0.1;
    } 

    // ---------------------------------------------------------------------
    // Create and add lights into the scene, call only after startup()
    addLights() {
        const color = 0xFFFFFF;
        const intensity1 = 1;
        const light1 = new THREE.DirectionalLight(color, intensity1);
        light1.position.set(-1, 2, 4);
        const intensity2 = 1;
        const light2 = new THREE.DirectionalLight(color, intensity2);
        light2.position.set(0, 2, 0);

        this.#scene.add(light1);
        this.#scene.add(light2); 
    }

    // ---------------------------------------------------------------------


    // ---------------------------------------------------------------------
    // Jude if render area needs to be resized and re-rendered 
    resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const pixelRatio = window.devicePixelRatio;
        const width = canvas.clientWidth * pixelRatio | 0;
        const height = canvas.clientHeight * pixelRatio | 0;
        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
            renderer.setSize(width, height, false);
        }
        return needResize;
    }

    // ---------------------------------------------------------------------
    // Update the controls, cameras, or the shaders 
    update() {

        var rotation = this.#camera.rotation; 
        
        this.#camera.position.set(-12, -5, -70);
        //-12.532438819599603, y: -5.712040425508086, z: -71.42744566067631
        this.#camera.rotation.set(3, 0, 2);
        //console.log(this.#camera.rotation);

    }

    // ---------------------------------------------------------------------
    // Examine the render area and redraw the canvas  
    redraw = (time) => {
        if (this.resizeRendererToDisplaySize(this.#renderer)) {
            const canvas = this.#renderer.domElement;
            this.#camera.aspect = canvas.clientWidth / canvas.clientHeight;
            this.#camera.updateProjectionMatrix();
        }

        

        requestAnimationFrame(this.redraw); // Context is automatically preserved

        // Manual movement rely on hard coded mouse movement detection 
        if (this.#manualMovement) {
            this.update(); 
        } else {
            this.#controls.update();
            //console.log(this.#camera.rotation)
            //console.log(this.#camera.position)
        }

        this.#renderer.render(this.#scene, this.#camera);
    }


    // ---------------------------------------------------------------------
    // Main function of the class 
    main() {

        this.startup();
        this.addLights();
        this.loadGeometriesRock(); 

        requestAnimationFrame(this.redraw);
    }

}


