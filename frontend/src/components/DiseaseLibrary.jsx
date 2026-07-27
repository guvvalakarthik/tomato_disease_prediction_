import { DISEASES, SEVERITY_STYLES } from "../diseases.js";

export default function DiseaseLibrary() {
  return (
    <section id="diseases" className="px-4 sm:px-6 pb-24 scroll-mt-24">
      <div className="max-w-6xl mx-auto">
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold text-slate-900">Disease Library</h2>
          <p className="mt-3 text-slate-600">
            The model recognizes these 10 tomato leaf conditions.
          </p>
        </div>

        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(DISEASES).map(([key, d]) => (
            <div
              key={key}
              className="bg-white rounded-2xl border border-slate-200 p-6 hover:shadow-lg hover:-translate-y-0.5 transition-all"
            >
              <div className="flex items-center justify-between">
                <span className="text-3xl">{d.emoji}</span>
                <span
                  className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${SEVERITY_STYLES[d.severity]}`}
                >
                  {d.severity === "none" ? "HEALTHY" : d.severity.toUpperCase()}
                </span>
              </div>
              <h3 className="mt-4 font-bold text-slate-900">{d.name}</h3>
              <p className="mt-2 text-sm text-slate-600 leading-relaxed">
                {d.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
