import { useAppStore } from '../../stores/useAppStore';
import type { TabType } from '../../types';
import styles from './TabBar.module.css';

const TABS: { key: TabType; label: string }[] = [
  { key: 'markdown', label: 'Markdown' },
  { key: 'chemical', label: 'Chemical Elements' },
  { key: 'json', label: 'JSON' },
];

export default function TabBar() {
  const activeTab = useAppStore(s => s.activeTab);
  const setActiveTab = useAppStore(s => s.setActiveTab);

  return (
    <div className={styles.tabBar}>
      {TABS.map(tab => (
        <button
          key={tab.key}
          className={`${styles.tab} ${activeTab === tab.key ? styles.active : ''}`}
          onClick={() => setActiveTab(tab.key)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
