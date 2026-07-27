import Navbar from "./components/Navbar.jsx";
import Hero from "./components/Hero.jsx";
import UploadSection from "./components/UploadSection.jsx";
import DiseaseLibrary from "./components/DiseaseLibrary.jsx";
import Footer from "./components/Footer.jsx";

export default function App() {
  return (
    <div className="min-h-screen bg-white text-slate-900">
      <Navbar />
      <main>
        <Hero />
        <UploadSection />
        <DiseaseLibrary />
      </main>
      <Footer />
    </div>
  );
}
