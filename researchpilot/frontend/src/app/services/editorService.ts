export function getWsUrl(workspaceId: string) {
  const token = localStorage.getItem("access_token");
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/workspace/${workspaceId}?token=${token}`;
}

export async function fetchDocument(docId: number) {
  const token = localStorage.getItem("access_token");
  const res = await fetch(`/api/editor/doc/${docId}`, { headers: { Authorization: token ? `Bearer ${token}` : "" } });
  if (!res.ok) throw new Error("Document fetch failed");
  return res.json();
}
