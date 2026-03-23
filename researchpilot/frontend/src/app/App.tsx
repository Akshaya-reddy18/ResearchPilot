import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { LandingPage } from './components/research/LandingPage';
import { ResearchDashboard } from './components/research/ResearchDashboard';
import { UploadPDF } from './components/research/UploadPDF';
import { ChatInterface } from './components/research/ChatInterface';
import { WorkspacesPage } from './components/research/WorkspacesPage';
import { WorkspaceDetailPage } from './components/research/WorkspaceDetailPage';
import { EditorPage } from './components/research/EditorPage';
import { Sidebar } from './components/research/Sidebar';
import { LoginPage } from './components/research/LoginPage';
import { SignupPage } from './components/research/SignupPage';
import { ProtectedRoute } from './components/ProtectedRoute';

function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      {children}
    </div>
  );
}

function EditorLayout({ children }: { children: React.ReactNode }) {
  // Editor page does not show the left Sidebar column — full-width editor
  return <div className="flex h-screen overflow-hidden w-full">{children}</div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Public */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />

        {/* Dashboard */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <ResearchDashboard />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* Upload */}
        <Route
          path="/upload"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <UploadPDF />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* Workspaces list */}
        <Route
          path="/workspaces"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <WorkspacesPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* 🔥 NEW: Workspace detail */}
        <Route
          path="/workspace/:id"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <WorkspaceDetailPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* 🔥 NEW: Editor page */}
        <Route
          path="/doc/:id"
          element={
            <ProtectedRoute>
              <EditorLayout>
                <EditorPage />
              </EditorLayout>
            </ProtectedRoute>
          }
        />

        {/* Chat */}
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <ChatInterface />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />

      </Routes>
    </BrowserRouter>
  );
}