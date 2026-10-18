import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { Alert, LoadingState } from "../components/ui";

const PatientForm = lazy(() => import("../components/PatientForm").then((module) => ({ default: module.PatientForm })));
const DashboardPage = lazy(() => import("../pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const PatientDetailPage = lazy(() => import("../pages/PatientDetailPage").then((module) => ({ default: module.PatientDetailPage })));
const PatientsPage = lazy(() => import("../pages/PatientsPage").then((module) => ({ default: module.PatientsPage })));

function NotFoundPage() {
  return <Alert tone="warning" title="Página não encontrada">O endereço solicitado não corresponde a uma página disponível.</Alert>;
}

export function AppRoutes() {
  return (
    <Suspense fallback={<LoadingState label="Carregando página..." />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/patients" element={<PatientsPage />} />
          <Route path="/patients/new" element={<PatientForm />} />
          <Route path="/patients/:patientId" element={<PatientDetailPage />} />
          <Route path="/not-found" element={<NotFoundPage />} />
          <Route path="*" element={<Navigate to="/not-found" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
