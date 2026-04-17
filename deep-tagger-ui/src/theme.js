import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#7b3ff2',
      light: '#a56cf7',
      dark: '#5a1fc4',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#b388ff',
      light: '#d7b8ff',
      dark: '#805acb',
      contrastText: '#ffffff',
    },
    background: {
      default: '#faf7ff',
      paper: '#ffffff',
    },
  },
  shape: {
    borderRadius: 10,
  },
});

export default theme;
