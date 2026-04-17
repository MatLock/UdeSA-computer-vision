import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import './Loading.css';

function Loading({ label = 'Loading...', size = 56 }) {
  return (
    <Box className="loading-container">
      <CircularProgress color="primary" size={size} thickness={4} />
      {label && (
        <Typography variant="body2" color="text.secondary" className="loading-label">
          {label}
        </Typography>
      )}
    </Box>
  );
}

export default Loading;
