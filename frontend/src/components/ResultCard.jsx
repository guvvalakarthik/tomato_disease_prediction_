import { Stethoscope } from "lucide-react";
import { DISEASES, SEVERITY_STYLES } from "../diseases.js";

export default function ResultCard({ result }) {
  const info = DISEASES[result.class] || {
    name: result.class,
    severity: "medium",
    emoji: "🍅",
    description: "",
    treatment: "",
  };
  const confidence = (result.confidence * 100).toFixed(1);
  const healthy = result.class === "Tomato_healthy";

  return (
    <div className="mt-8 animate-fade-up">
      <div
        className={`rounded-2xl border p-6 ${
          healthy
            ? "bg-emerald-50 border-emerald-200"
            : "bg-orange-50/60 border-orange-200"
        }`}
      >
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-xs uppercase tracking-wide font-semibold text-slate-500">
              Diagnosis
            </p>
            <h3 className="mt-1 text-2xl font-bold text-slate-900 flex items-center gap-2">
              <span>{info.emoji}</span> {info.name}
            </h3>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-bold border ${SEVERITY_STYLES[info.severity]}`}
          >
            {healthy ? "HEALTHY" : `${info.severity.toUpperCase()} SEVERITY`}
          </span>
        </div>

        <div className="mt-5">
          <div className="flex justify-between text-sm font-medium text-slate-600 mb-1.5">
            <span>Confidence</span>
            <span className="font-bold text-slate-900">{confidence}%</span>
          </div>
          <div className="h-3 rounded-full bg-white border border-slate-200 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                healthy ? "bg-emerald-500" : "bg-orange-500"
              }`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        {info.description && (
          <p className="mt-5 text-sm text-slate-600 leading-relaxed">
            {info.description}
          </p>
        )}

        {info.treatment && (
          <div className="mt-4 flex gap-3 bg-white rounded-xl border border-slate-200 p-4">
            <Stethoscope size={18} className="text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-slate-800">
                {healthy ? "Care tips" : "Recommended treatment"}
              </p>
              <p className="mt-1 text-sm text-slate-600 leading-relaxed">
                {info.treatment}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
