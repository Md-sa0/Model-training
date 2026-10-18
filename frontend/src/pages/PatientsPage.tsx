import { Plus } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toApiError } from "../api/client";
import { PatientsTable } from "../components/PatientsTable";
import { Button, Card } from "../components/ui";
import { usePatients } from "../hooks/useApi";

export function PatientsPage() {
  const [offset, setOffset] = useState(0);
  const patients = usePatients(offset);
  const navigate = useNavigate();
  const error = patients.error ? toApiError(patients.error) : null;

  return (
    <div className="page-stack">
      <Card
        title="Pacientes"
        description="Identificadores retornados pelo banco local; nenhum dado demonstrativo é inserido."
        action={<Button onClick={() => navigate("/patients/new")}><Plus size={17} /> Novo paciente</Button>}
      >
        <PatientsTable
          patients={patients.data?.items}
          total={patients.data?.total}
          offset={offset}
          isLoading={patients.isPending}
          isError={patients.isError}
          errorMessage={error?.message}
          onPageChange={setOffset}
          onRetry={() => void patients.refetch()}
        />
      </Card>
    </div>
  );
}
