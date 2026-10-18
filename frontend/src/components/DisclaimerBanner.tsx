import { Info } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="persistent-disclaimer" role="note">
      <Info size={17} aria-hidden="true" />
      <p>
        <strong>Modelo experimental, não validado para decisões clínicas.</strong>{" "}
        A API registra dados enviados manualmente; isso não é monitoramento em tempo real.
        O resultado não substitui avaliação profissional.
      </p>
    </div>
  );
}
