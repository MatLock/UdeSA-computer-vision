import React from 'react';
import { Snackbar, Alert } from '@mui/material';
import './Toast.css';

function Toast({ open, message, severity = 'error', onClose, autoHideDuration = 4000 }) {
  return (
    <Snackbar
      open={open}
      autoHideDuration={autoHideDuration}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      className="toast"
    >
      <Alert
        onClose={onClose}
        severity={severity}
        variant="filled"
        className="toast-alert"
        size="small"
      >
        {message}
      </Alert>
    </Snackbar>
  );
}

export default Toast;
