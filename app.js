require('dotenv').config() // load env vars from .env file
const express = require('express');
const { Canvas, Image, ImageData, createCanvas, loadImage } = require('canvas');
const faceapi = require('face-api.js');
const axios = require('axios');
const https = require('https');
const bodyParser = require('body-parser');
const cache = new Map();
const app = express();
const PORT = process.env.PORT || 3000;

// Define a threshold for determining if the images are of the same person
const TRESHOLD = process.env.TRESHOLD || 0.6;

// Middleware to parse JSON bodies
app.use(express.json({limit: '10mb'}));

// Middleware to parse x-www-form-urlencoded bodies
app.use(bodyParser.urlencoded({limit: '10mb', extended: true }));

// Set up face-api.js
const { CanvasRenderingContext2D } = Canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, CanvasRenderingContext2D });

// Load face-api.js model
async function loadModels() {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromDisk('models'),
        faceapi.nets.faceLandmark68Net.loadFromDisk('models'),
        faceapi.nets.faceRecognitionNet.loadFromDisk('models')
    ]);
    console.log('Models loaded successfully');
}
loadModels().catch(error => console.error('Error loading models:', error));

// Create an HTTPS agent with rejectUnauthorized set to false
const httpsAgent = new https.Agent({ rejectUnauthorized: false });

// Function to detect face from image buffer
async function detectFace(imageBuffer) {
    if(cache.has(imageBuffer)) {
        return imageBuffer;
    }
    // Convert image buffer to Image object
    const img = new Image();
    img.src = imageBuffer;

    // Detect face from image
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
    cache.set(imageBuffer, detection);
    return detection;
}

// Function to detect face from image buffer
async function detectFaceFromBase64(base64Data) {
    if(cache.has(base64Data)) {
        return cache.get(base64Data);
    }
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
            try {
                // Convert image to canvas
                const canvas = createCanvas(img.width, img.height);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, img.width, img.height);

                // Detect face from canvas
                const detection = await faceapi.detectSingleFace(canvas).withFaceLandmarks().withFaceDescriptor();
                resolve(detection);
                cache.set(base64Data, detection);
            } catch (error) {
                reject(error);
            }
        };
        img.onerror = reject;
        img.src = `${base64Data}`;
    });
}

async function compareFaces(face1, face2) {
    if (!face1 || !face2) {
        // If either image doesn't contain a face, return false
        return { match: null, samePerson: false };
    } else {
        // Calculate distance between face descriptors
        const distance = faceapi.euclideanDistance(face1.descriptor, face2.descriptor);

        // Determine if the images are of the same person
        const samePerson = distance < TRESHOLD;

        // If the images are of the same person, return the match result
        // Otherwise, return false
        return { match: samePerson ? 'Match' : 'No match', samePerson };
    }
}

// Compare faces route with URLs
app.post('/compare-faces/url', async (req, res) => {
    try {
        const { selfie: selfieUrl, id_photo: idPhotoUrl } = req.body;

        // Fetch images from URLs using axios with custom HTTPS agent
        const [selfieRes, idPhotoRes] = await Promise.all([
            axios.get(selfieUrl, { responseType: 'arraybuffer', httpsAgent }),
            axios.get(idPhotoUrl, { responseType: 'arraybuffer', httpsAgent })
        ]);

        if (!selfieRes || !idPhotoRes || selfieRes.status !== 200 || idPhotoRes.status !== 200) {
            throw new Error('Failed to fetch images');
        }

        // Detect faces in images
        const face1 = await detectFace(Buffer.from(selfieRes.data, 'binary'));
        const face2 = await detectFace(Buffer.from(idPhotoRes.data, 'binary'));

        // call compare faces function
        const obj_res = await compareFaces(face1, face2);
        res.json(obj_res);

    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'An error occurred' });
    }
});

// Compare faces route with base64 images
app.post('/compare-faces/base64', async (req, res) => {
    try {
        const { selfie: selfieBase64, id_photo: idPhotoBase64 } = req.body;

        // Decode base64 images
        //const selfieBuffer = Buffer.from(selfieBase64, 'base64');
        //const idPhotoBuffer = Buffer.from(idPhotoBase64, 'base64');

        const face1 = await detectFaceFromBase64(selfieBase64);
        const face2 = await detectFaceFromBase64(idPhotoBase64);

        // Detect faces in images
        //const face1 = await detectFace(selfieBuffer);
        //const face2 = await detectFace(idPhotoBuffer);

        // call compare faces function
        const obj_res = await compareFaces(face1, face2);
        res.json(obj_res);

    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'An error occurred' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});