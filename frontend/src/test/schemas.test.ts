import { describe, expect, it } from "vitest";
import { buildObservationPayload, observationFormSchema, patientFormSchema, toPatientPayload } from "../lib/schemas";
import { featureKeys } from "../types/api";

const emptyFeatureValues = Object.fromEntries(featureKeys.map((key) => [key, ""])) as Record<string, string>;

describe("schemas dos formulários", () => {
  it("rejeita campos clínicos não numéricos", () => {
    const result = observationFormSchema.safeParse({
      episode_id: "f53c9c1f-c024-45f4-8da1-a12e67408619",
      observed_at: "2025-04-12T08:30",
      values: { ...emptyFeatureValues, HR: "rápido" },
      units: emptyFeatureValues,
    });

    expect(result.success).toBe(false);
    if (!result.success) expect(result.error.issues[0].path).toEqual(["values", "HR"]);
  });

  it("converte campos em branco para ausentes e inclui data com fuso", () => {
    const input = {
      episode_id: "f53c9c1f-c024-45f4-8da1-a12e67408619",
      observed_at: "2025-04-12T08:30",
      values: { ...emptyFeatureValues, HR: "82.5" },
      units: { ...emptyFeatureValues, HR: "bpm" },
    };
    const parsed = observationFormSchema.parse(input);
    const payload = buildObservationPayload(parsed, "d937d2c0-c6bb-4f64-9a55-3bba6e81d399");

    expect(payload.values.HR).toBe(82.5);
    expect(payload.values.Temp).toBeNull();
    expect(payload.units).toEqual({ HR: "bpm" });
    expect(payload.observed_at).toMatch(/(?:Z|[+-]\d{2}:\d{2})$/);
  });

  it("omite campos opcionais vazios no cadastro e normaliza identificador", () => {
    const input = patientFormSchema.parse({
      external_id: "  demo-123  ",
      initial_episode_label: "",
      episode_started_at: "",
    });

    expect(toPatientPayload(input)).toEqual({ external_id: "demo-123" });
  });
});
