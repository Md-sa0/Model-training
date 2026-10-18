import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "../api/client";
import { createAssessment, getObservations, getPatients } from "../api/endpoints";

describe("cliente HTTP", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("envia parâmetros de paginação para a lista de pacientes e devolve a resposta", async () => {
    const response = { items: [], total: 0, limit: 25, offset: 50 };
    const get = vi.spyOn(apiClient, "get").mockResolvedValue({ data: response });

    await expect(getPatients(25, 50)).resolves.toEqual(response);
    expect(get).toHaveBeenCalledWith("/api/patients", { params: { limit: 25, offset: 50 } });
  });

  it("filtra observações pelo episódio solicitado", async () => {
    const response = { items: [], total: 0, limit: 10, offset: 0 };
    const get = vi.spyOn(apiClient, "get").mockResolvedValue({ data: response });

    await expect(getObservations("patient-id", "episode-id", 10, 0)).resolves.toEqual(response);
    expect(get).toHaveBeenCalledWith("/api/patients/patient-id/observations", {
      params: { episode_id: "episode-id", limit: 10, offset: 0 },
    });
  });

  it("cria avaliação associada à observação e retorna a resposta do servidor", async () => {
    const result = { assessment_id: "assessment-id", observation_id: "observation-id" };
    const post = vi.spyOn(apiClient, "post").mockResolvedValue({ data: result });

    await expect(createAssessment("patient-id", {
      assessment_id: "assessment-id",
      observation_id: "observation-id",
    })).resolves.toEqual(result);
    expect(post).toHaveBeenCalledWith("/api/patients/patient-id/assessments", {
      assessment_id: "assessment-id",
      observation_id: "observation-id",
    });
  });
});
