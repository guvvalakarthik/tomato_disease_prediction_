import { DISEASES, SEVERITY_STYLES } from "../diseaseCatalog.js";

export default function SafeDiseaseLibrary() {
  return (
    <section id="diseases" className="px-4 sm:px-6 pb-24 scroll-mt-24">
      <div className="max-w-6xl mx-auto">
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold text-slate-900">Supported classes</h2>
          <p className="mt-3 text-slate-600">
            Nine disease or pest classes and one healthy class. Similar symptoms can have different causes.
          </p>
        </div>

        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(DISEASES).map(([key, disease]) => (
            <article
              key={key}
              className="bg-white rounded-2xl border border-slate-200 p-6 hover:shadow-lg hover:-translate-y-0.5 transition-all"
            >
              <div className="flex items-center justify-between">
                <span className="grid place-items-center h-10 min-w-10 px-2 rounded-lg bg-slate-100 text-sm font-bold text-slate-700">
                  {disease.emoji}
                </span>
                <span
                  className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${SEVERITY_STYLES[disease.severity]}`}
                >
                  {disease.severity === "none" ? "HEALTHY" : disease.severity.toUpperCase()}
                </span>
              </div>
              <h3 className="mt-4 font-bold text-slate-900">{disease.name}</h3>
              <p className="mt-2 text-sm text-slate-600 leading-relaxed">
                {disease.description}
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
