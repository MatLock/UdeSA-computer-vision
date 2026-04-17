import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Box, Typography, Button, TextField, Stack } from '@mui/material';
import Toast from '../../components/Toast/Toast';
import './Landing.css';

const URL_REGEX = /^https?:\/\/([\w-]+\.)+[a-z]{2,}(:\d+)?(\/[^\s]*)?$/i;

function Landing() {
  const navigate = useNavigate();
  const [imageUrl, setImageUrl] = useState('');
  const [toastOpen, setToastOpen] = useState(false);

  const handleStart = () => {
    const value = imageUrl.trim();
    if (!URL_REGEX.test(value)) {
      setToastOpen(true);
      return;
    }
    navigate('/results', { state: { imageUrl: value } });
  };

  return (
    <Container maxWidth="md" className="landing-container">
      <Box className="landing-header">
        <Typography variant="h5" component="h1" className="landing-title">
          Deep Tagger
        </Typography>
      </Box>

      <Box className="landing-center">
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          alignItems="center"
          className="landing-form"
        >
          <TextField
            label="Image URL"
            variant="outlined"
            fullWidth
            value={imageUrl}
            onChange={(e) => setImageUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleStart()}
          />
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={handleStart}
            className="landing-submit"
          >
            Start Prediction
          </Button>
        </Stack>
      </Box>

      <Toast
        open={toastOpen}
        message="Please provide a valid URL"
        severity="error"
        onClose={() => setToastOpen(false)}
      />
    </Container>
  );
}

export default Landing;
