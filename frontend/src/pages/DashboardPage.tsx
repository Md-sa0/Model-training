import { ArrowRight, Plus, RefreshCw, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toApiError } from "../api/client";
import { ConnectionPanel } from "../components/ConnectionPanel";
import { DisclaimerBanner } from "../components/DisclaimerBanner";
import { PatientsTable } from "../components/PatientsTable";
import { Button, Card } from "../components/ui";
import { usePatients } from "../hooks/useApi";
import { useState } from "react";

export function DashboardPage() {
  const [offset, setOffset] = useState(0);
  const patients = usePatients(offset);
  const navigate = useNavigate();
  const error = patients.error ? toApiError(patients.error) : null;

  return (
    <div className="page-stack">
      <DisclaimerBanner />
      <ConnectionPanel />
      <Card
        title="Pacientes cadastrados"
        description="Registros locais retornados pela API. Não representa uma população hospitalar."
        action={
          <Button onClick={() => navigate("/patients/new")}>
            <Plus size={17} /> Cadastrar paciente
          </Button>
        }
      >
        <div className="list-context">
          <span><Users size={16} /> {patients.data?.total ?? "—"} registro(s) no banco local</span>
          <Button variant="ghost" onClick={() => void patients.refetch()} aria-label="Atualizar lista de pacientes">
            <RefreshCw size={15} /> Atualizar
          </Button>
        </div>
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
        {patients.data?.items.length ? (
          <button className="text-link" onClick={() => navigate("/patients")}>
            Abrir lista de pacientes <ArrowRight size={15} />
          </button>
        ) : null}
      </Card>
    </div>
  );
}
