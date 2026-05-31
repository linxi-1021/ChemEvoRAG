import { useState, useCallback } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import styles from './PageNavigator.module.css';

interface Props {
  totalPages: number;
}

export default function PageNavigator({ totalPages }: Props) {
  const currentPage = useAppStore(s => s.currentPage);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const scale = useAppStore(s => s.scale);
  const setScale = useAppStore(s => s.setScale);
  const [inputValue, setInputValue] = useState('');

  const handlePrev = useCallback(() => {
    setCurrentPage(Math.max(0, currentPage - 1));
  }, [currentPage, setCurrentPage]);

  const handleNext = useCallback(() => {
    setCurrentPage(Math.min(totalPages - 1, currentPage + 1));
  }, [currentPage, totalPages, setCurrentPage]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputCommit = () => {
    const page = parseInt(inputValue, 10);
    if (!isNaN(page) && page >= 1 && page <= totalPages) {
      setCurrentPage(page - 1);
    }
    setInputValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleInputCommit();
    }
  };

  return (
    <div className={styles.navigator}>
      <div className={styles.pageControls}>
        <button
          className={styles.btn}
          onClick={handlePrev}
          disabled={currentPage <= 0}
          title="Previous page"
        >
          ◀
        </button>
        <div className={styles.pageInfo}>
          <input
            className={styles.pageInput}
            type="text"
            value={inputValue || String(currentPage + 1)}
            onChange={handleInputChange}
            onBlur={handleInputCommit}
            onKeyDown={handleKeyDown}
          />
          <span className={styles.pageTotal}>/ {totalPages}</span>
        </div>
        <button
          className={styles.btn}
          onClick={handleNext}
          disabled={currentPage >= totalPages - 1}
          title="Next page"
        >
          ▶
        </button>
      </div>

      <div className={styles.zoomControls}>
        <button
          className={styles.btn}
          onClick={() => setScale(scale - 0.25)}
          disabled={scale <= 0.5}
          title="Zoom out"
        >
          −
        </button>
        <span className={styles.zoomLabel}>{Math.round(scale * 100)}%</span>
        <button
          className={styles.btn}
          onClick={() => setScale(scale + 0.25)}
          disabled={scale >= 3}
          title="Zoom in"
        >
          +
        </button>
      </div>
    </div>
  );
}
