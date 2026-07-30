import { AlertTriangle, Info, ScanSearch } from "lucide-react";
import { DISEASES, SEVERITY_STYLES } from "../diseaseCatalog.js";

export default function PredictionResult({ result }) {
  const uncertain = result.status === "uncertain";
  const primary = result.prediction;
  const info = primary ? DISEASES[primary.class_id] : null;
  const healthy = primary?.class_id === "tomato_healthy";

  return (
    <section
      aria-live="polite"
      className={`mt-8 rounded-2xl border p-6 animate-fade-up ${
        uncertain
          ? "bg-amber-50 border-amber-300"
          : healthy
            ? "bg-emerald-50 border-emerald-200"
            : "bg-orange-50/60 border-orange-200"
      }`}
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="text-xs uppercase tracking-wide font-semibold text-slate-500">
            Screening result
          </p>
          <h3 className="mt-1 text-2xl font-bold text-slate-900 flex items-center gap-2">
            {uncertain ? (
              <><AlertTriangle size={24} /> Unable to diagnose confidently</>
            ) : (
              <>{info?.name || primary.label}</>
            )}
          </h3>
        </div>
        {info && (
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold border ${SEVERITY_STYLES[info.severity]}`}
          >
            {healthy ? "HEALTHY" : `${info.severity.toUpperCase()} PRIORITY`}
          </span>
        )}
      </div>

      {uncertain ? (
        <p className="mt-4 text-sm text-slate-700 leading-relaxed">
          {result.uncertainty_reason}
        </p>
      ) : (
        <>
          <div className="mt-5">
            <div className="flex justify-between text-sm font-medium text-slate-600 mb-1.5">
              <span>Calibrated confidence</span>
              <span className="font-bold text-slate-900">
                {(primary.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-3 rounded-full bg-white border border-slate-200 overflow-hidden">
              <div
                className={`h-full rounded-full ${healthy ? "bg-emerald-500" : "bg-orange-500"}`}
                style={{ width: `${primary.probability * 100}%` }}
              />
            </div>
          </div>
          <p className="mt-5 text-sm text-slate-600 leading-relaxed">
            {info?.description}
          </p>
          <div className="mt-4 flex gap-3 bg-white rounded-xl border border-slate-200 p-4">
            <Info size={18} className="text-emerald-700 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-slate-800">General next step</p>
              <p className="mt-1 text-sm text-slate-600 leading-relaxed">{info?.guidance}</p>
              <p className="mt-2 text-xs text-slate-500">
                General references: {" "}
                <a className="underline hover:text-emerald-700" href="https://extension.umn.edu/vegetables/disease-management" target="_blank" rel="noreferrer">University of Minnesota Extension</a>
                {" ? "}
                <a className="underline hover:text-emerald-700" href="https://ipm.ucanr.edu/agriculture/tomato/" target="_blank" rel="noreferrer">UC Integrated Pest Management</a>
              </p>
            </div>
          </div>
        </>
      )}

      <div className="mt-5 border-t border-slate-200 pt-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Top classes</p>
        <ol className="mt-2 space-y-1">
          {result.top_predictions.map((item) => (
            <li key={item.class_id} className="flex justify-between text-sm text-slate-700">
              <span>{item.label}</span>
              <span>{(item.probability * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ol>
      </div>

      {result.explanation && (
        <figure className="mt-5 rounded-xl bg-white border border-slate-200 p-4">
          <figcaption className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <ScanSearch size={17} /> Model attention map
          </figcaption>
          <img
            className="mt-3 w-full max-h-72 object-contain rounded-lg [image-rendering:auto]"
            src={`data:image/png;base64,${result.explanation.png_base64}`}
            alt="Class activation map showing image regions used by the model"
          />
          <p className="mt-2 text-xs text-slate-500">Attention is diagnostic evidence, not proof of causality.</p>
        </figure>
      )}

      <p className="mt-5 text-xs text-slate-500 leading-relaxed">{result.disclaimer}</p>
      <p className="mt-1 text-xs text-slate-400">Model {result.model.version}</p>
    </section>
  );
}
