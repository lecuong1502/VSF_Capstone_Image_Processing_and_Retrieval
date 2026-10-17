import ReviewPage from "./review/ReviewPage";

function App() {
  if (window.location.pathname.startsWith("/review")) return <ReviewPage />;
  return (
    <main className="home">
      <h1>VSF – ảnh du lịch Việt Nam</h1>
      <p>
        Giao diện tìm kiếm sẽ có ở #27. Công cụ kiểm duyệt:{" "}
        <a href="/review">/review</a>
      </p>
    </main>
  );
}

export default App;
