import { useCallback, useRef, useState } from "react";
import axios from "axios";
import { UploadCloud, ImageIcon, Loader2, RotateCcw } from "lucide-react";
import PredictionResult from "./PredictionResult.jsx";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png"]);

export default function UploadSection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const selectFile = useCallback((f) => {
    if (!f || !ACCEPTED_TYPES.has(f.type)) {
      setError("Please select a JPEG or PNG image.");
      return;
    }
    if (f.size > MAX_FILE_BYTES) {
      setError("The image must be 4 MB or smaller.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files[0]);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const predict = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post(`${API_URL}/v1/predict?explain=true`, formData, { timeout: 90000 });
      setResult(data);
    } catch (requestError) {
      const message = requestError.response?.data?.message;
      setError(message || "The screening service is unavailable. A free-tier cold start can take about one minute; please retry.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="detect" className="px-4 sm:px-6 pb-20 scroll-mt-24">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-3xl border border-slate-200 shadow-xl shadow-emerald-900/5 p-6 sm:p-10">
          <h2 className="text-2xl font-bold text-slate-900 text-center">
            Analyze a Leaf
          </h2>
          <p className="mt-1 text-sm text-slate-500 text-center">
            Drag & drop or browse — JPG and PNG supported
          </p>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`mt-6 rounded-2xl border-2 border-dashed cursor-pointer transition-all grid place-items-center min-h-56 p-6 ${
              dragging
                ? "border-emerald-500 bg-emerald-50"
                : "border-slate-300 hover:border-emerald-400 hover:bg-emerald-50/50"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              onChange={(e) => selectFile(e.target.files[0])}
            />
            {preview ? (
              <img
                src={preview}
                alt="Leaf preview"
                className="max-h-72 rounded-xl object-contain shadow-md"
              />
            ) : (
              <div className="text-center text-slate-500">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-emerald-100 text-emerald-600 grid place-items-center mb-4">
                  <UploadCloud size={26} />
                </div>
                <p className="font-semibold text-slate-700">
                  Drop your leaf image here
                </p>
                <p className="text-sm mt-1">or click to browse files</p>
              </div>
            )}
          </div>

          {error && (
            <p className="mt-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-3">
              {error}
            </p>
          )}

          <div className="mt-6 flex gap-3">
            <button
              onClick={predict}
              disabled={!file || loading}
              className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-emerald-600 text-white font-semibold hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" /> Analyzing…
                </>
              ) : (
                <>
                  <ImageIcon size={18} /> Detect Disease
                </>
              )}
            </button>
            {file && (
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 px-5 py-3.5 rounded-xl border border-slate-300 text-slate-600 font-semibold hover:bg-slate-50 transition-colors"
              >
                <RotateCcw size={16} /> Reset
              </button>
            )}
          </div>

          {result && <PredictionResult result={result} />}
        </div>
      </div>
    </section>
  );
}
