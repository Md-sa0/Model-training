import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useCreateEpisode } from "../hooks/useApi";
import { toApiError } from "../api/client";
import { episodeFormSchema, toEpisodePayload } from "../lib/schemas";
import type { z } from "zod";
import { Alert, Button, Field, TextInput } from "./ui";

type EpisodeFields = z.infer<typeof episodeFormSchema>;

export function EpisodeForm({ patientId, onCreated }: { patientId: string; onCreated: (episodeId: string) => void }) {
  const createEpisode = useCreateEpisode(patientId);
  const form = useForm<EpisodeFields>({
    resolver: zodResolver(episodeFormSchema),
    defaultValues: { label: "", started_at: "" },
  });

  async function submit(values: EpisodeFields) {
    try {
      const created = await createEpisode.mutateAsync(toEpisodePayload(values));
      onCreated(created.episode_id);
      form.reset();
    } catch {
      // Mutation error is rendered below.
    }
  }

  const error = createEpisode.error ? toApiError(createEpisode.error) : null;

  return (
    <form className="form-stack" onSubmit={form.handleSubmit(submit)} noValidate>
      {error && <Alert tone="error" title="Não foi possível criar o episódio">{error.message}</Alert>}
      <Field label="Rótulo do episódio (opcional)" htmlFor="episode-label" error={form.formState.errors.label?.message}>
        <TextInput id="episode-label" maxLength={200} {...form.register("label")} />
      </Field>
      <Field
        label="Data e hora de início (opcional)"
        htmlFor="episode-start"
        hint="Se omitida, o backend registra o horário atual UTC."
      >
        <TextInput id="episode-start" type="datetime-local" {...form.register("started_at")} />
      </Field>
      <Button type="submit" disabled={createEpisode.isPending}>
        {createEpisode.isPending ? "Criando..." : "Criar episódio"}
      </Button>
    </form>
  );
}
