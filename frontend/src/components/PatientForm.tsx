import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowLeft, UserRoundPlus } from "lucide-react";
import { useForm } from "react-hook-form";
import { Link, useNavigate } from "react-router-dom";
import { useCreatePatient } from "../hooks/useApi";
import { toApiError } from "../api/client";
import { patientFormSchema, toPatientPayload } from "../lib/schemas";
import { formatDate } from "../lib/format";
import type { z } from "zod";
import { Alert, Button, Card, Field, TextInput } from "./ui";

type PatientFields = z.infer<typeof patientFormSchema>;

export function PatientForm() {
  const navigate = useNavigate();
  const createPatient = useCreatePatient();
  const form = useForm<PatientFields>({
    resolver: zodResolver(patientFormSchema),
    defaultValues: { external_id: "", initial_episode_label: "", episode_started_at: "" },
  });

  async function onSubmit(values: PatientFields) {
    try {
      const created = await createPatient.mutateAsync(toPatientPayload(values));
      navigate(`/patients/${created.patient.patient_id}?episode_id=${created.episode.episode_id}`, {
        state: { createdPatient: created },
      });
    } catch {
      // Mutation error is rendered below.
    }
  }

  const apiError = createPatient.error ? toApiError(createPatient.error) : null;

  return (
    <div className="narrow-page">
      <Link className="back-link" to="/"><ArrowLeft size={16} /> Voltar à visão geral</Link>
      <Card title="Cadastrar paciente" description="O backend criará também o episódio inicial na mesma operação.">
        {apiError && (
          <Alert tone="error" title={apiError.status === 409 ? "Identificador já cadastrado" : "Não foi possível cadastrar"}>
            {apiError.message}
          </Alert>
        )}
        <form className="form-stack" onSubmit={form.handleSubmit(onSubmit)} noValidate>
          <Field
            label="Identificador externo (opcional)"
            htmlFor="external_id"
            error={form.formState.errors.external_id?.message}
            hint="Deve ser um identificador não pessoal apropriado ao seu ambiente."
          >
            <TextInput id="external_id" maxLength={200} autoComplete="off" {...form.register("external_id")} />
          </Field>
          <Field
            label="Rótulo do episódio inicial (opcional)"
            htmlFor="initial_episode_label"
            error={form.formState.errors.initial_episode_label?.message}
          >
            <TextInput id="initial_episode_label" maxLength={200} {...form.register("initial_episode_label")} />
          </Field>
          <Field
            label="Início do episódio (opcional)"
            htmlFor="episode_started_at"
            error={form.formState.errors.episode_started_at?.message}
            hint="O horário informado será convertido para UTC. Se omitido, o backend usa o horário atual UTC."
          >
            <TextInput id="episode_started_at" type="datetime-local" {...form.register("episode_started_at")} />
          </Field>
          {createPatient.data && (
            <Alert tone="success" title="Paciente cadastrado">
              Episódio inicial {createPatient.data.episode.episode_id} criado em {formatDate(createPatient.data.episode.started_at)}.
            </Alert>
          )}
          <div className="form-actions">
            <Button type="submit" disabled={createPatient.isPending}>
              <UserRoundPlus size={17} />
              {createPatient.isPending ? "Cadastrando..." : "Cadastrar paciente"}
            </Button>
            <Link className="button button-ghost" to="/">Cancelar</Link>
          </div>
        </form>
      </Card>
    </div>
  );
}
