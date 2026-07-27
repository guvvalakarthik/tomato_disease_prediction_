import { ScanSearch, ShieldCheck, Zap } from "lucide-react";

const stats = [
  { icon: ScanSearch, label: "10 disease classes" },
  { icon: Zap, label: "Instant AI results" },
  { icon: ShieldCheck, label: "Treatment guidance" },
];

export default function Hero() {
  return (
    <section className="relative pt-32 pb-20 px-4 sm:px-6 overflow-hidden">
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-emerald-50 via-white to-white" />
      <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-emerald-200/40 blur-3xl -z-10" />
      <div className="absolute top-40 -left-32 w-80 h-80 rounded-full bg-red-200/30 blur-3xl -z-10" />

      <div className="max-w-3xl mx-auto text-center animate-fade-up">
        <span className="inline-block px-4 py-1.5 rounded-full bg-emerald-100 text-emerald-700 text-sm font-semibold mb-6">
          🍅 AI-Powered Plant Health
        </span>
        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-slate-900">
          Detect tomato diseases
          <span className="block bg-gradient-to-r from-emerald-600 to-teal-500 bg-clip-text text-transparent">
            in seconds
          </span>
        </h1>
        <p className="mt-6 text-lg text-slate-600 max-w-xl mx-auto">
          Upload a photo of a tomato leaf and our deep learning model will
          identify the disease and suggest a treatment plan.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {stats.map(({ icon: Icon, label }) => (
            <div
              key={label}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-emerald-100 shadow-sm text-sm font-medium text-slate-700"
            >
              <Icon size={16} className="text-emerald-600" />
              {label}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
