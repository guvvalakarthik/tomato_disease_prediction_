import Navbar from "./components/Navbar.jsx";
import FieldHero from "./components/FieldHero.jsx";
import UploadSection from "./components/UploadSection.jsx";
import SafeDiseaseLibrary from "./components/SafeDiseaseLibrary.jsx";
import SiteFooter from "./components/SiteFooter.jsx";

export default function App() {
  return (
    <div className="min-h-screen bg-white text-slate-900">
      <Navbar />
      <main>
        <FieldHero />
        <UploadSection />
        <SafeDiseaseLibrary />
      </main>
      <SiteFooter />
    </div>
  );
}
