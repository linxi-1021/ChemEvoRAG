import { useAppStore } from './stores/useAppStore';
import { useDocumentData } from './hooks/useDocumentData';
import DocumentList from './components/sidebar/DocumentList';
import PdfViewer from './components/pdf/PdfViewer';
import TabBar from './components/tabs/TabBar';
import MarkdownTab from './components/markdown/MarkdownTab';
import ChemicalTab from './components/chemical/ChemicalTab';
import JsonTab from './components/json/JsonTab';
import styles from './App.module.css';

function App() {
  const selectedDocId = useAppStore(s => s.selectedDocId);
  const activeTab = useAppStore(s => s.activeTab);
  const docData = useDocumentData();

  return (
    <div className={styles.app}>
      {/* Left Sidebar */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h1 className={styles.logo}>ChemEvoRAG</h1>
          <span className={styles.logoSub}>Parser Viewer</span>
        </div>
        <DocumentList />
      </aside>

      {/* Center: PDF Viewer */}
      <main className={styles.center}>
        {selectedDocId ? (
          <PdfViewer docId={selectedDocId} docData={docData} />
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📄</div>
            <h2>Select a document</h2>
            <p>Choose a paper from the left sidebar to view its parsing results.</p>
          </div>
        )}
      </main>

      {/* Right: Results Panel */}
      <aside className={styles.rightPanel}>
        {selectedDocId && (
          <>
            <TabBar />
            <div className={styles.tabContent}>
              {activeTab === 'markdown' && (
                <MarkdownTab docId={selectedDocId} docData={docData} />
              )}
              {activeTab === 'chemical' && (
                <ChemicalTab docId={selectedDocId} docData={docData} />
              )}
              {activeTab === 'json' && (
                <JsonTab docId={selectedDocId} docData={docData} />
              )}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

export default App;
