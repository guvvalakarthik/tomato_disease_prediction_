import "./ImageUpload.css";
import "./Landing.css";

import React, { useState } from "react";
import axios from "axios";

export function ImageUpload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult("");
    setConfidence("");
  };

  const handleUpload = async () => {
      console.log("Button clicked"); 
    if (!image) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data.class);
      setConfidence((response.data.confidence * 100).toFixed(2));
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Check backend.");
    } finally {
      setLoading(false);
    }
  };
  return (
  <div className="background">
    <div className="overlay"></div>

    <div className="glass-card">
      <div className="title">🍅 Tomato Disease Detection</div>

      <div className="upload-box">
        <input
          type="file"
          onChange={handleChange}
          style={{ display: "none" }}
          id="fileInput"
        />
        <label htmlFor="fileInput" style={{ cursor: "pointer" }}>
          Drag & Drop or Click to Upload Leaf Image
        </label>
      </div>

      {preview && (
        <img
          src={preview}
          alt="Preview"
          style={{ width: "100%", marginTop: "20px", borderRadius: "10px" }}
        />
      )}

      <button onClick={handleUpload} className="button">
        Predict
      </button>

      {result && (
        <div className="result">
          <h3>Disease: {result}</h3>
          <h4>Confidence: {confidence}%</h4>
        </div>
      )}
    </div>
  </div>
);

}

