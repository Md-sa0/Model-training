import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DashboardPage } from "../pages/DashboardPage";
import { PatientForm } from "../components/PatientForm";
import { PatientDetailPage } from "../pages/PatientDetailPage";
import * as api from "../api/endpoints";

vi.mock("../api/endpoints", () => ({
  getHealth: vi.fn(),
  getConfig: vi.fn(),
  getPatients: vi.fn(),
  createPatient: vi.fn(),
  getPatient: vi.fn(),
  getEpisodes: vi.fn(),
  getObservations: vi.fn(),
  getAssessments: vi.fn(),
  createAssessment: vi.fn(),
  createEpisode: vi.fn(),
  createObservation: vi.fn(),
}));

const patientId = "5a38c360-8afe-4930-9ff6-b6fafb60fc6e";
const episodeId = "249488c7-bbc6-4c94-83bd-d5fa763497bb";
const observationId = "c0d4a250-f50d-4b93-9f3a-bf091193b44c";
const emptyPage = { items: [], total: 0, limit: 20, offset: 0 };
const patient = { patient_id: patientId, external_id: "demo-123", created_at: "2025-04-12T08:00:00Z" };
const episode = {
  episode_id: episodeId,
  patient_id: patientId,
  label: "Internação inicial",
  started_at: "2025-04-12T08:00:00Z",
  created_at: "2025-04-12T08:00:00Z",
};

function renderWithProviders(ui: React.ReactNode, initialEntry = "/") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

function detailRoutes() {
  return (
    <Routes>
      <Route path="/patients/new" element={<PatientForm />} />
      <Route path="/patients/:patientId" element={<PatientDetailPage />} />
    </Routes>
  );
}

describe("fluxos da aplicação", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getHealth).mockResolvedValue({ status: "ok", model_ready: true, model_error: null });
    vi.mocked(api.getConfig).mockResolvedValue({
      model_ready: true,
      model_id: "qwen",
      model_version: "test",
      config_version: "config-test",
      calibration_method: "isotonic",
      calibrator_locked: true,
      threshold: 0.01,
      threshold_locked: true,
      threshold_target_met: false,
      fitted_on: null,
      test_used_for_fitting: false,
      features: [],
      dynamic_features: [],
      decision_semantics: "binary",
      risk_score_meaning: "experimental score",
      disclaimer: "experimental",
    });
    vi.mocked(api.getPatients).mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
    vi.mocked(api.getPatient).mockResolvedValue({ ...patient, episodes: [episode] });
    vi.mocked(api.getEpisodes).mockResolvedValue([episode]);
    vi.mocked(api.getObservations).mockResolvedValue(emptyPage);
    vi.mocked(api.getAssessments).mockResolvedValue(emptyPage);
  });

  it("exibe carregamento, erro e estado vazio da listagem", async () => {
    let resolvePatients!: (value: Awaited<ReturnType<typeof api.getPatients>>) => void;
    vi.mocked(api.getPatients).mockReturnValueOnce(new Promise((resolve) => { resolvePatients = resolve; }));
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText("Carregando pacientes cadastrados...")).toBeInTheDocument();
    resolvePatients({ items: [], total: 0, limit: 50, offset: 0 });
    expect(await screen.findByText("Nenhum paciente cadastrado")).toBeInTheDocument();

    vi.mocked(api.getPatients).mockRejectedValueOnce(Object.assign(new Error("API offline"), {
      isAxiosError: true,
      response: { status: 500, data: { detail: "API offline" } },
    }));
    renderWithProviders(<DashboardPage />);
    expect(await screen.findByText("API offline")).toBeInTheDocument();

    vi.mocked(api.getPatients).mockResolvedValueOnce({ items: [], total: 0, limit: 50, offset: 0 });
    renderWithProviders(<DashboardPage />);
    expect(await screen.findByText("Nenhum paciente cadastrado")).toBeInTheDocument();
  });

  it("cadastra paciente e abre o episódio inicial retornado pela API", async () => {
    vi.mocked(api.createPatient).mockResolvedValue({ patient, episode });
    renderWithProviders(detailRoutes(), "/patients/new");

    fireEvent.change(screen.getByLabelText("Identificador externo (opcional)"), { target: { value: "demo-123" } });
    fireEvent.click(screen.getByRole("button", { name: "Cadastrar paciente" }));

    expect(await screen.findByText("Identificação do paciente")).toBeInTheDocument();
    expect(screen.getByLabelText("Episódio ativo para consulta")).toHaveValue(episodeId);
    expect(screen.getByText(episodeId)).toBeInTheDocument();
    expect(api.createPatient).toHaveBeenCalledWith({ external_id: "demo-123" });
    await waitFor(() => expect(api.getObservations).toHaveBeenCalledWith(patientId, episodeId, 20, 0));
  });

  it("comunica explicitamente quando a inferência retorna 503", async () => {
    vi.mocked(api.getObservations).mockResolvedValue({
      items: [{
        observation_id: observationId,
        patient_id: patientId,
        episode_id: episodeId,
        observed_at: "2025-04-12T08:30:00Z",
        received_at: "2025-04-12T08:31:00Z",
        values: {},
        units: {},
        missing_fields: ["HR"],
      }],
      total: 1,
      limit: 20,
      offset: 0,
    });
    vi.mocked(api.createAssessment).mockRejectedValue(Object.assign(new Error("not ready"), {
      isAxiosError: true,
      response: { status: 503, data: { detail: { code: "MODEL_NOT_READY", message: "model not ready" } } },
    }));
    renderWithProviders(detailRoutes(), `/patients/${patientId}?episode_id=${episodeId}`);

    fireEvent.click(await screen.findByRole("button", { name: /Solicitar avaliação para observação/ }));
    expect(await screen.findByText("Modelo indisponível para inferência")).toBeInTheDocument();
    expect(screen.getByText(/503 MODEL_NOT_READY/)).toBeInTheDocument();
    expect(api.createAssessment).toHaveBeenCalledWith(patientId, {
      assessment_id: expect.any(String),
      observation_id: observationId,
    });
  });
});
