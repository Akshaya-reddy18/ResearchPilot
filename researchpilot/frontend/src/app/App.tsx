import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { LandingPage } from './components/research/LandingPage';
import { ResearchDashboard } from './components/research/ResearchDashboard';
import { UploadPDF } from './components/research/UploadPDF';
import { ChatInterface } from './components/research/ChatInterface';
import { WorkspacesPage } from './components/research/WorkspacesPage';
import { Sidebar } from './components/research/Sidebar';

function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      {children}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/dashboard"
          element={
            <DashboardLayout>
              <ResearchDashboard />
            </DashboardLayout>
          }
        />
        <Route
          path="/upload"
          element={
            <DashboardLayout>
              <UploadPDF />
            </DashboardLayout>
          }
        />
        <Route
          path="/workspaces"
          element={
            <DashboardLayout>
              <WorkspacesPage />
            </DashboardLayout>
          }
        />
        <Route
          path="/chat"
          element={
            <DashboardLayout>
              <ChatInterface />
            </DashboardLayout>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
